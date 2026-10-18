"""端点一致性核查：官方 `deepseek-flash` 与聚合端点的 `deepseek-v4.1-flash` 是不是同一个模型？

为什么这件事对本项目重要
    整个项目的定位声明是"三个模型经**第三方聚合端点**接入，真实后端不可见，
    结论仅作方法学演示"。如果聚合端点的 `deepseek-v4.1-flash` 与官方
    `deepseek-flash`（官方口径下 V4.1-Flash 的正式名）行为一致，
    那条免责声明就可以收紧为"至少 DeepSeek 模型已与官方端点交叉验证"。

三类证据，按**可信度从高到低**

① **分词器指纹（最强）**：同一段 prompt，`prompt_tokens` 必须**逐题完全相同**。
   分词器是确定性的，不受采样随机性影响 —— 这是唯一不受"模型本身不确定"干扰的证据。
   ⚠️ 前提：必须**匹配思考开关**。实测官方端点在开启思考时会注入额外的 prompt 前缀
   （同一段文本 prompt_tokens 从 72 变成 98），不匹配就会把"思考开关差异"
   误读成"分词器差异"。

② **答案一致率（中等）**：跨端点一致率 vs **同端点自一致率**（噪声地板）。
   ★ 这条证据的能力**有上限**：C3 实测聚合端点的自一致率只有 72%，
   所以两个"不同但相似"的模型也可能拿到 60–70% 的跨端点一致率。
   只有当跨端点一致率**明显低于**自一致率时，它才能证伪"同一模型"。

③ **准确率与行为分布（弱）**：同 100 题上的准确率、延迟、reasoning 长度、
   finish_reason 分布。这些只能作为旁证。

用法
    .venv\\Scripts\\python.exe scripts\\compare_endpoints.py --n-reps 3
    .venv\\Scripts\\python.exe scripts\\compare_endpoints.py --report-only   # 只复算

产物
    runs/{run_id}/raw/c1/{official_model}.jsonl
    runs/{run_id}/endpoint_comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode, PromptSpec  # noqa: E402

OFFICIAL_MODEL_DEFAULT = "deepseek-flash"
AGG_ARM_DEFAULT = "deepseek-v4.1-flash"


def _hash_of(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_agg_rows(run_id: str, arm: str, cfg: Config) -> list:
    """聚合端点侧的原始记录（复用已跑完的 run，不重复调用）。"""
    return [r for r in execute.load_results(run_id, cfg) if r.arm == arm]


def official_params() -> dict:
    """与聚合端点那条模型**匹配**的参数。

    聚合端点的 deepseek 模型冻结的是 max_tokens=1024 / temperature=0.1 /
    reasoning_effort=none。官方端点实测同样认 `reasoning_effort`，
    因此可以对齐思考开关 —— 这是①类证据成立的前提。
    """
    _cfg, arms = adapt.load_arms()
    arm = next((a for a in arms if a.model == AGG_ARM_DEFAULT), None)
    p = dict(arm.params) if arm else {"max_tokens": 1024, "temperature": 0.1}
    p.pop("timeout_s", None)
    p.pop("max_retries", None)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="端点一致性核查")
    ap.add_argument("--agg-run", default="c3-full", help="聚合端点侧已有 run（默认 c3-full）")
    ap.add_argument("--agg-arm", default=AGG_ARM_DEFAULT)
    ap.add_argument("--official-model", default=OFFICIAL_MODEL_DEFAULT)
    ap.add_argument("--n-reps", type=int, default=3, help="官方端点每题重复次数")
    ap.add_argument("--run-id", default="endpoint-cmp")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    tmpl = (paths.ROOT / "prompts" / "c1_strict.txt").read_text(encoding="utf-8")
    agg_rows = load_agg_rows(args.agg_run, args.agg_arm, Config.C3)
    if not agg_rows:
        print(f"!! {args.agg_run} 里没有 {args.agg_arm} 的原始记录")
        return 2
    inst_ids = sorted({r.instance_id for r in agg_rows})

    print("=" * 78)
    print("端点一致性核查")
    print("=" * 78)
    print(f"  聚合端点 : {args.agg_run} / {args.agg_arm}（{len(inst_ids)} 题，已跑完，复用）")
    print(f"  官方端点 : {paths.judge_base_url()} / {args.official_model}（{args.n_reps} 次/题）")

    # ---- 官方端点跑批 ------------------------------------------------------
    params = official_params()
    print(f"  匹配参数 : {params}")
    specs = [
        PromptSpec(run_id=args.run_id, config=Config.C3, arm=args.official_model,
                   instance_id=iid, rep=rep,
                   prompt_text=adapt.render_prompt(pool[iid], tmpl), params=dict(params))
        for iid in inst_ids for rep in range(args.n_reps)
    ]
    stats = None
    if not args.report_only:
        stats = execute.run_specs(
            run_id=args.run_id, specs=specs,
            arms_concurrency={args.official_model: args.concurrency}, limit=args.limit,
        )
        print(f"  执行     : 成功 {int(stats['executed']) - int(stats['failed'])}｜"
              f"失败 {stats['failed']}｜错误 {stats['by_error_kind']}")
    # ★ manifest **总是**写（含 --report-only）：它的作用是留下 provenance，
    #   与"这次有没有发请求"无关。之前把它放在跑批分支里，
    #   结果 report-only 补跑一次仍然没有 manifest，门禁一直 WARN。
    # ★ configs 刻意留空：这是**辅助探针**，不是配置矩阵里的一次跑批。
    #   若声明 configs=["c3"]，门禁会按 matrix.yaml 里 c3 的 reps=10 去算期望条数
    #   （100×10=1000），而这里实际只有 100×3=300 —— 会误报"raw 不完整"。
    execute.write_manifest(
        run_id=args.run_id, configs=[],
        arms=[{"name": args.official_model, "params": params}],
        data_hash=_hash_of(paths.data() / "eval_set.jsonl"),
        models_snapshot=[args.official_model],
        note=f"端点一致性核查（辅助探针）｜官方 {args.official_model} vs "
             f"聚合 {args.agg_arm}｜聚合侧复用 {args.agg_run}｜"
             f"{len(inst_ids)} 题 × {args.n_reps} 次 = {len(specs)} 条",
        stats=stats,
    )

    off = [r for r in execute.load_results(args.run_id, Config.C3)
           if r.arm == args.official_model]
    if not off:
        print("!! 没有官方端点记录")
        return 2

    # ---- ① 分词器指纹 ------------------------------------------------------
    agg_pin = {r.instance_id: r.tokens_in for r in agg_rows if r.rep == 0}
    off_pin = {r.instance_id: r.tokens_in for r in off if r.rep == 0}
    common = sorted(set(agg_pin) & set(off_pin))
    diffs = [off_pin[i] - agg_pin[i] for i in common]
    exact = sum(1 for d in diffs if d == 0)
    tok = {
        "n_compared": len(common),
        "n_exact_match": exact,
        "match_rate": round(exact / len(common), 4) if common else None,
        "diff_mean": round(statistics.mean(diffs), 3) if diffs else None,
        "diff_min": min(diffs) if diffs else None,
        "diff_max": max(diffs) if diffs else None,
        "agg_prompt_tokens_total": sum(agg_pin[i] for i in common),
        "off_prompt_tokens_total": sum(off_pin[i] for i in common),
    }

    # ---- ② 答案一致率 ------------------------------------------------------
    def answers(rows) -> dict[str, list[int | None]]:
        g: dict[str, list[int | None]] = defaultdict(list)
        for r in rows:
            idx, _ = tally.parse_answer(r.content, pool[r.instance_id].n_options,
                                        ParseMode.STRICT)
            g[r.instance_id].append(idx)
        return g

    a_ans, o_ans = answers(agg_rows), answers(off)
    ids = sorted(set(a_ans) & set(o_ans))

    def self_same(g: dict) -> float:
        """**全部** rep 完全一致的比例（严格口径）。"""
        n = ok = 0
        for i in ids:
            clean = [x for x in g[i] if x is not None]
            if not clean:
                continue
            n += 1
            ok += int(len(set(clean)) == 1)
        return round(ok / n, 4) if n else float("nan")

    def self_pairwise(g: dict) -> float:
        """同一端点内、**逐对** rep 相同的比例。

        ★ 这才是与"跨端点逐对一致率"可比的噪声地板。
          `self_same`（全部 rep 都相同）是严得多的口径：
          若某题答案分布为 {A:0.8, B:0.2}，逐对一致率约 0.68，
          而"10 次全同"只有 0.8^10+0.2^10 ≈ 0.107。
          拿这两个数直接相减会得出错误的结论（本脚本第一版就是这么错的）。
        """
        eq = n = 0
        for i in ids:
            vals = [x for x in g[i] if x is not None]
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    n += 1
                    eq += int(vals[a] == vals[b])
        return round(eq / n, 4) if n else float("nan")

    def modal(v: list) -> int | None:
        clean = [x for x in v if x is not None]
        if not clean:
            return None
        c = Counter(clean).most_common()
        return c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else None

    pair_eq = pair_n = 0
    for i in ids:
        for x in a_ans[i]:
            for y in o_ans[i]:
                if x is None or y is None:
                    continue
                pair_n += 1
                pair_eq += int(x == y)
    modal_eq = modal_n = 0
    for i in ids:
        ma, mo = modal(a_ans[i]), modal(o_ans[i])
        if ma is None or mo is None:
            continue
        modal_n += 1
        modal_eq += int(ma == mo)

    agree = {
        "agg_self_same_rate": self_same(a_ans),
        "official_self_same_rate": self_same(o_ans),
        "agg_self_pairwise_rate": self_pairwise(a_ans),
        "official_self_pairwise_rate": self_pairwise(o_ans),
        "cross_pairwise_equal_rate": round(pair_eq / pair_n, 4) if pair_n else None,
        "cross_pairwise_n": pair_n,
        "cross_modal_equal_rate": round(modal_eq / modal_n, 4) if modal_n else None,
        "cross_modal_n": modal_n,
    }
    # 可比的噪声地板：两个端点各自"逐对一致率"的均值
    floor = None
    if agree["agg_self_pairwise_rate"] == agree["agg_self_pairwise_rate"] \
            and agree["official_self_pairwise_rate"] == agree["official_self_pairwise_rate"]:
        floor = round((agree["agg_self_pairwise_rate"]
                       + agree["official_self_pairwise_rate"]) / 2, 4)
    agree["self_pairwise_floor"] = floor
    agree["cross_minus_floor"] = (round((agree["cross_pairwise_equal_rate"] or 0) - floor, 4)
                                  if floor is not None else None)

    # 分词器指纹不一致的题目（要能点名，否则"96% 一致"没法解释）
    tok_mismatch = sorted(
        ({"instance_id": i, "agg": agg_pin[i], "official": off_pin[i],
          "diff": off_pin[i] - agg_pin[i]} for i in common if off_pin[i] != agg_pin[i]),
        key=lambda d: d["diff"],
    )
    tok["mismatches"] = tok_mismatch[:20]

    # ---- ③ 准确率与行为 ----------------------------------------------------
    def acc_of(rows) -> dict:
        tiers = []
        for r in rows:
            idx, _ = tally.parse_answer(r.content, pool[r.instance_id].n_options,
                                        ParseMode.STRICT)
            if idx is None:
                tiers.append("invalid")
            else:
                tiers.append("correct" if idx in pool[r.instance_id].gold_indices else "wrong")
        return metrics.tier_accounting(tiers)

    def behav(rows) -> dict:
        lat = sorted(r.latency_s for r in rows)
        rl = sorted(len(r.reasoning) for r in rows)
        return {
            "n": len(rows),
            "median_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
            "p95_latency_s": round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))], 3) if lat else None,
            "median_reasoning_len": rl[len(rl) // 2] if rl else None,
            "finish_reasons": dict(Counter(r.finish_reason for r in rows)),
            "resp_model_echo": dict(Counter(r.resp_model for r in rows)),
        }

    acc = {"aggregator": acc_of(agg_rows), "official": acc_of(off)}
    bhv = {"aggregator": behav(agg_rows), "official": behav(off)}

    # ---- 结论 --------------------------------------------------------------
    mr = tok["match_rate"] or 0
    if mr >= 0.98:
        tok_verdict = "分词器指纹**逐题一致** ⇒ 与「同一分词器 + 同一 prompt 模板」相符"
    elif mr >= 0.5:
        tok_verdict = "分词器指纹**部分一致** ⇒ 可能版本不同，或存在前缀差异"
    else:
        tok_verdict = "分词器指纹**基本不一致** ⇒ 几乎可以确定不是同一部署"

    print()
    print("=" * 78)
    print("① 分词器指纹（最强证据，不受采样随机性影响）")
    print("=" * 78)
    print(f"  逐题 prompt_tokens 完全相同 : {tok['n_exact_match']}/{tok['n_compared']}"
          f"  ({mr:.1%})")
    print(f"  差值 均值/最小/最大         : {tok['diff_mean']} / {tok['diff_min']} / {tok['diff_max']}")
    print(f"  两侧 prompt_tokens 合计     : {tok['agg_prompt_tokens_total']} vs {tok['off_prompt_tokens_total']}")
    print(f"  判定：{tok_verdict}")
    if tok["mismatches"]:
        print(f"  不一致的 {len(tok['mismatches'])} 题（要能点名，否则"
              f"「{mr:.0%} 一致」无法解释）：")
        for d in tok["mismatches"][:8]:
            print(f"    {d['instance_id']:<18} 聚合 {d['agg']:>4}  官方 {d['official']:>4}  "
                  f"差 {d['diff']:+d}")

    print()
    print("=" * 78)
    print("② 答案一致率（受噪声地板限制，只能证伪不能证实）")
    print("=" * 78)
    print("  [严格口径] 全部 rep 都相同")
    print(f"    聚合端点（10 次）          : {agree['agg_self_same_rate']:.1%}")
    print(f"    官方端点（3 次）           : {agree['official_self_same_rate']:.1%}")
    print("  [可比口径] 逐对相同 ← 只有这一组能互相比较")
    print(f"    聚合端点内部               : {agree['agg_self_pairwise_rate']:.1%}")
    print(f"    官方端点内部               : {agree['official_self_pairwise_rate']:.1%}")
    print(f"    跨端点                     : {agree['cross_pairwise_equal_rate']:.1%}"
          f"  (n={agree['cross_pairwise_n']})")
    print(f"    跨端点众数一致率           : {agree['cross_modal_equal_rate']:.1%}"
          f"  (n={agree['cross_modal_n']})")
    if agree["cross_minus_floor"] is not None:
        print(f"  ★ 跨端点 − 各自内部（噪声地板 {agree['self_pairwise_floor']:.1%}）: "
              f"{agree['cross_minus_floor']:+.1%}")
    print("    读法：≈0 或为正 ⇒「跨端点差异」不大于「同端点重复差异」，与同一模型相符；")
    print("          显著为负 ⇒ 跨端点差异更大，与不同模型相符。")
    print("    ⚠️ 这条只能**证伪**、不能**证实**：两个高度相似的模型同样会给出高一致率。")

    print()
    print("=" * 78)
    print("③ 准确率与行为分布（旁证）")
    print("=" * 78)
    for side in ("aggregator", "official"):
        a, b = acc[side], bhv[side]
        print(f"  {side:<12} acc={metrics.format_acc(a['acc'])} "
              f"(可判 {a['n_gradeable']}/{a['n_total']})  "
              f"中位延迟 {b['median_latency_s']}s  reasoning中位 {b['median_reasoning_len']}")
        print(f"  {'':<12} finish={b['finish_reasons']}  resp_model={b['resp_model_echo']}")

    out = paths.ROOT / "runs" / args.run_id / "endpoint_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "aggregator": {"run": args.agg_run, "arm": args.agg_arm},
        "official": {"endpoint": paths.judge_base_url(), "model": args.official_model,
                     "params": params},
        "tokenizer_fingerprint": tok, "agreement": agree,
        "accuracy": acc, "behaviour": bhv,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
