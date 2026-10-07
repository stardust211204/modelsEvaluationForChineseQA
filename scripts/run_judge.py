"""C4 · LLM-as-Judge 复核：judge 独立判定 C1 的输出，与规则判分比对。

为什么 judge 必须走**独立端点**
    judge 与被评模型同源会引入 self-preference。本项目三臂都在聚合端点上，
    因此 judge 用 **DeepSeek 官方端点**的 `deepseek-v4-pro`。

    ⚠️ 但它仍与 `deepseek-v4.1-flash` **同家族**，违反 `configs/matrix.yaml`
       原先写的"须与被评模型不同家族"。处理方式不是回避，而是**分臂**报一致率：
       若 deepseek 臂的 judge-vs-规则一致率**显著高于**另两臂，那本身就是
       self-preference 的实测证据 —— 把混淆变成观测量。

为什么 judge 只作复核
    MCQ 有标准答案，规则判分可辩护；judge 存在 validity 争议。
    所以 judge 结论**不替换**规则判分，只回答"规则判分有没有系统性偏差"。

用法
    .venv\\Scripts\\python.exe scripts\\run_judge.py --source-run c1-full --limit 6   # 试跑
    .venv\\Scripts\\python.exe scripts\\run_judge.py --source-run c1-full            # 全量

产物
    runs/{run_id}/raw/c4/{judge_model}.jsonl     原始 judge 输出
    runs/{run_id}/tallies/c4/all.jsonl           judge 判定（arm = **被判的臂**）
    runs/{run_id}/judge_vs_rule.json             分臂一致率报告
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode, PromptSpec  # noqa: E402

JUDGE_PARAMS = {"max_tokens": 2048, "temperature": 0.1}


def _hash_of(p: Path) -> str:
    """与 run_config._hash_of 同一算法：sha256(原始字节)[:16]。"""
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_instances() -> dict:
    p = paths.data() / "eval_set.jsonl"
    return {i.id: i for i in build_set.load_frozen(p)}


def build_judge_specs(*, run_id: str, spec: adapt.ConfigSpec, pool: list,
                      source_run: str, source_cfg: Config, judge_model: str,
                      arms: list[str] | None, params: dict) -> tuple[list[PromptSpec], dict]:
    """为「每个 (题, 被判臂)」各造一条 judge 请求。"""
    tmpl = (paths.ROOT / spec.prompt_template).read_text(encoding="utf-8")
    instances = adapt.subset_from_c1(pool, spec.n_instances)
    ids = {i.id for i in instances}

    src = [r for r in execute.load_results(source_run, source_cfg)
           if r.instance_id in ids and r.rep == 0]
    if arms:
        src = [r for r in src if r.arm in arms]
    by_id = {i.id: i for i in instances}

    specs: list[PromptSpec] = []
    for r in src:
        inst = by_id[r.instance_id]
        specs.append(PromptSpec(
            run_id=run_id, config=Config.C4, arm=judge_model,
            instance_id=r.instance_id, rep=0,
            prompt_text=adapt.render_judge_prompt(inst, r.content, tmpl),
            params=dict(params),
            target_arm=r.arm,          # ★ 进去重键，否则三臂互相覆盖
        ))
    meta = {"n_instances": len(instances), "n_source_results": len(src),
            "judged_arms": sorted({r.arm for r in src})}
    return specs, meta


def judge_vs_rule(rule_rows: list[dict], judge_rows: list[dict], arm: str) -> dict:
    """一个臂的 judge-vs-规则一致率。

    参考 = 规则判分（严格口径），待验证 = judge。
    非二元档位（invalid/truncated）由 `binary_agreement` 自动排除并如实报出。
    """
    rule = {t["instance_id"]: t["tier"] for t in rule_rows
            if t["arm"] == arm and t["mode"] == ParseMode.STRICT.value}
    jd = {t["instance_id"]: t["tier"] for t in judge_rows if t["arm"] == arm}
    shared = sorted(set(rule) & set(jd))
    pairs = [(rule[i], jd[i]) for i in shared]
    labels = ["correct", "wrong", "invalid", "truncated"]
    out = {
        "n_shared": len(shared),
        "binary": metrics.binary_agreement(pairs),
        "all_tiers": metrics.confusion_matrix(pairs, labels),
        "rule_tier_counts": {k: sum(1 for i in shared if rule[i] == k) for k in labels},
        "judge_tier_counts": {k: sum(1 for i in shared if jd[i] == k) for k in labels},
        "judge_disagreements": [
            {"instance_id": i, "rule": rule[i], "judge": jd[i]}
            for i in shared if rule[i] != jd[i]
        ][:20],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="C4 judge 复核")
    ap.add_argument("--source-run", default="c1-full", help="被复核的 run（默认 c1-full）")
    ap.add_argument("--source-config", default="c1")
    ap.add_argument("--run-id", default="c4-full")
    ap.add_argument("--arms", default=None, help="逗号分隔，默认三臂全判")
    ap.add_argument("--limit", type=int, default=None, help="本次最多执行多少条（试跑）")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="judge 的 token 预算。★ 太小会让推理吃光预算、content 为空")
    ap.add_argument("--judge-param", action="append", default=None,
                    help="额外 judge 参数（可重复）：--judge-param reasoning_effort=none")
    ap.add_argument("--report-only", action="store_true", help="只出报告，不发请求")
    args = ap.parse_args()

    judge_params = {"max_tokens": args.max_tokens, "temperature": 0.1}
    for kv in (args.judge_param or []):
        if "=" not in kv:
            print(f"!! --judge-param 需要 KEY=VALUE 形式，收到：{kv}")
            return 2
        k, _, v = kv.partition("=")
        low = v.strip().lower()
        judge_params[k.strip()] = (low == "true" if low in ("true", "false") else v.strip())

    source_cfg = Config(args.source_config)
    pool = build_set.load_frozen(paths.data() / "eval_set.jsonl")
    spec = next(s for s in adapt.load_matrix() if s.key is Config.C4)

    judge_model = paths.judge_model()
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None

    print("=" * 78)
    print(f"C4 judge 复核   run_id={args.run_id}")
    print("=" * 78)
    print(f"  judge 端点  : {paths.judge_base_url()}")
    print(f"  judge 模型  : {judge_model}")
    print(f"  被复核 run  : {args.source_run}/{source_cfg.value}")
    print(f"  judge 参数  : {judge_params}")

    specs, meta = build_judge_specs(
        run_id=args.run_id, spec=spec, pool=pool, source_run=args.source_run,
        source_cfg=source_cfg, judge_model=judge_model, arms=arms,
        params=judge_params,
    )
    print(f"  被判臂      : {meta['judged_arms']}")
    print(f"  待执行      : {len(specs)} 条（{meta['n_instances']} 题 × {len(meta['judged_arms'])} 臂）")

    if not specs:
        print("!! 没有可判条目：检查 --source-run 是否有 raw")
        return 2

    if not args.report_only:
        stats = execute.run_specs(
            run_id=args.run_id, specs=specs,
            arms_concurrency={judge_model: args.concurrency}, limit=args.limit,
        )
        execute.write_manifest(
            run_id=args.run_id, configs=[Config.C4.value],
            arms=[{"name": judge_model, "params": JUDGE_PARAMS}],
            data_hash=_hash_of(paths.data() / "eval_set.jsonl"),
            models_snapshot=[judge_model],
            note=f"LLM-as-Judge 复核｜judge={judge_model}（官方端点）｜"
                 f"被复核={args.source_run}｜分臂报一致率以暴露 self-preference",
            stats=stats,
        )
        print(f"  失败        : {stats.get('failed')}")

    # ---- 判分（judge 输出 -> Tally，arm 记为**被判的臂**）------------------
    results = [r for r in execute.load_results(args.run_id, Config.C4)]
    if not results:
        print("!! 没有 judge 结果可判")
        return 2
    rows = [tally.tally_judge(r, r.target_arm or "unknown", Config.C4) for r in results]
    tpath = paths.tallies_dir(args.run_id, Config.C4.value) / "all.jsonl"
    tpath.write_text(
        "\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in rows) + "\n",
        encoding="utf-8",
    )
    judge_rows = [t.to_dict() for t in rows]

    # ---- 与规则判分比对 ----------------------------------------------------
    rule_rows = metrics.load_tallies(args.source_run, source_cfg, drop_excluded=False)  # 与 judge 输入对齐
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id, "judge_model": judge_model,
        "judge_endpoint": paths.judge_base_url(),
        "source_run": args.source_run, "arms": {},
        "note": "judge 与 deepseek-v4.1-flash 同家族；分臂报一致率以暴露 self-preference",
    }
    print()
    print("=" * 78)
    print("judge vs 规则判分（参考=规则严格口径，待验证=judge）")
    print("=" * 78)
    print(f"  {'被判臂':<28}{'共同题':>7}{'覆盖率':>8}{'一致率':>8}{'Cohen κ':>9}"
          f"{'judge判不出':>11}")
    for arm in sorted({t["arm"] for t in judge_rows}):
        rep = judge_vs_rule(rule_rows, judge_rows, arm)
        report["arms"][arm] = rep
        b = rep["binary"]
        if not b.get("n"):
            print(f"  {arm:<28}{rep['n_shared']:>7}{'—':>8}   无二元可比条目")
            continue
        print(f"  {arm:<28}{rep['n_shared']:>7}{b['coverage']:>8.0%}{b['accuracy']:>8.1%}"
              f"{(b['cohen_kappa'] or 0):>9.3f}{rep['judge_tier_counts']['invalid']:>11}")

    print()
    print("  读法：")
    print("   * 一致率高 = 规则判分与 judge 结论相符，规则判分可辩护。")
    print("   * **重点看臂间差异**：deepseek 臂明显更高 → 可能是 self-preference 的实测证据。")
    print("   * judge 判不出的条目（invalid）不算一致，也不该被算成答错。")

    out = paths.ROOT / "runs" / args.run_id / "judge_vs_rule.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Tally -> {tpath.relative_to(paths.ROOT)}")
    print(f"  JSON  -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
