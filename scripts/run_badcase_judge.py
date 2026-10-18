"""用 LLM 当 badcase 归因的**第二标注者**（官方 `deepseek-v4-pro`）。

目标
    人工只标一轮时没有一致率证据，归因分布不可引用（见 docs/badcase_method.md）。
    用 LLM 独立标一遍，就能算出 **人机一致率 + κ**，让分布变得可用。

为什么跑 2 遍（★ 关键设计，不是浪费）
    κ 的分母是「人 vs LLM」。若 LLM **自己两次都只有 60% 一致**，
    那么 κ=0.5 说明的是**任务噪声大**，而不是标签定得差。
    没有这个噪声地板，κ 无法解读 —— 与本项目 C3 用重复噪声做基准是同一条纪律。

避开本项目踩过的坑（每条都有前车之鉴）
    * **隐藏思考吃光预算**：C4 首轮 41/450 空输出，且失败**集中在最需要判的条目上**
      ⇒ 选择性偏差。故 `reasoning_effort=none`。
    * **不给 judge 看启发式归因**：已实测它只是"回复长度带"，给它看会把 judge 锚定到噪声。
    * **截断/空回复不进判定**：读不出内容就无从归因。
    * **解析取最后一个 `归因：`**：judge 可能在「理由」里复述格式。
    * **同家族偏差**：judge 是 deepseek 家族，而 104 条里有 24 条来自 deepseek 模型
      ⇒ 一律**按模型报告**。

用法
    .venv\\Scripts\\python.exe scripts\\run_badcase_judge.py --limit 6      # 试跑
    .venv\\Scripts\\python.exe scripts\\run_badcase_judge.py                # 全量（208 次）
    .venv\\Scripts\\python.exe scripts\\run_badcase_judge.py --report-only  # 只出报告
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, client, execute, paths  # noqa: E402
from meval.types import Config, ParseMode, PromptSpec, Tier  # noqa: E402

PROMPT_ATTR = "prompts/judge_badcase.txt"
PROMPT_GOLD = "prompts/judge_gold_audit.txt"
LETTERS = "ABCDEFGHIJKLMNOP"
N_REPS = 2

ATTR_LABELS = ["knowledge_gap", "reasoning_error", "option_confusion", "question_misread",
               "detail_omission", "format_parse", "cannot_tell"]
GOLD_LABELS = ["无", "D1", "D2", "D3"]

VERDICT_RE = re.compile(
    r"归因\s*[:：]\s*[`*]*\s*(" + "|".join(ATTR_LABELS) + r")\b", re.IGNORECASE)
DEFECT_RE = re.compile(r"缺陷\s*[:：]\s*[`*]*\s*(无|D1|D2|D3)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """取**最后一个** `归因：` —— judge 可能在「理由」里复述格式。"""
    ms = list(VERDICT_RE.finditer(text or ""))
    return ms[-1].group(1).lower() if ms else "UNCLEAR"


def parse_defect(text: str) -> str:
    """取**最后一个** `缺陷：`。"""
    ms = list(DEFECT_RE.finditer(text or ""))
    return ms[-1].group(1).upper() if ms else "UNCLEAR"


def load_human(run_id: str) -> dict[tuple[str, str], str]:
    """读人工标签。优先 `reports/*labels*.csv`（合并计划里的权威路径）。"""
    for d in (paths.ROOT / "reports", paths.ROOT / "runs" / run_id):
        for f in sorted(d.glob("*labels*.csv")):
            out: dict[tuple[str, str], str] = {}
            with f.open("r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    v = (row.get("confirmed_attribution") or "").strip()
                    iid = (row.get("instance_id") or "").strip()
                    arm = (row.get("arm") or "").strip()
                    if v and iid and arm:
                        out[(iid, arm)] = v
            if out:
                print(f"  人工标签 <- {f.relative_to(paths.ROOT)}（{len(out)} 条）")
                return out
    print("  ⚠️ 没找到人工标签（人机一致率将无法计算）")
    return {}


def kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return agree, ((agree - pe) / (1 - pe) if pe < 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description="badcase 归因 · LLM 第二标注者")
    ap.add_argument("--run-id", default="c2-full")
    ap.add_argument("--config", default="c2")
    ap.add_argument("--out-run", default="badcase-judge")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = Config(args.config)
    review = [json.loads(l) for l in
              (paths.ROOT / "runs" / args.run_id / "badcase_review.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    res = {(r.arm, r.instance_id): r for r in execute.load_results(args.run_id, cfg)}

    judge_model = paths.judge_model()
    base = paths.judge_base_url()
    tmpl = (paths.ROOT / PROMPT_ATTR).read_text(encoding="utf-8")

    print("=" * 78)
    print(f"badcase 归因 · LLM 第二标注者   out_run={args.out_run}")
    print("=" * 78)
    print(f"  judge 端点 : {base}")
    print(f"  judge 模型 : {judge_model}")
    print(f"  待标条目   : {len(review)}（每条跑 {args.reps} 遍测自一致率）")

    # ---- 组装（剔除读不出内容的）----------------------------------------
    specs, skipped = [], []
    for it in review:
        iid, arm = it["instance_id"], it["arm"]
        inst, m = pool.get(iid), res.get((arm, iid))
        if inst is None or m is None:
            skipped.append((iid, arm, "缺数据")); continue
        body = (m.content or "").strip()
        if m.finish_reason == "length" or not body:
            skipped.append((iid, arm, "截断/空回复")); continue
        idx, _ = __import__("meval.tally", fromlist=["x"]).parse_answer(
            body, len(inst.options), ParseMode.LENIENT)
        if idx is None:
            skipped.append((iid, arm, "判不出选择")); continue
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(inst.options))
        prompt = (tmpl.replace("{question}", inst.question)
                      .replace("{options}", opts)
                      .replace("{gold}", "".join(LETTERS[g] for g in inst.gold_indices))
                      .replace("{model_choice}", LETTERS[idx])
                      .replace("{reasoning}", body))
        for rep in range(args.reps):
            specs.append(PromptSpec(run_id=args.out_run, config=Config.C4,
                                    arm=judge_model, instance_id=iid, rep=rep,
                                    prompt_text=prompt,
                                    params={"max_tokens": 512, "temperature": 0.1,
                                            "reasoning_effort": "none"},
                                    target_arm=arm))
    print(f"  可标条目   : {len(specs) // args.reps}"
          f"｜跳过 {len(skipped)}"
          + (f"（{dict(Counter(s[2] for s in skipped))}）" if skipped else ""))
    print(f"  待执行     : {len(specs)} 次")

    if not args.report_only:
        cl = client.ChatClient(base_url=base, api_key=paths.judge_api_key())
        stats = execute.run_specs(run_id=args.out_run, specs=specs,
                                  arms_concurrency={judge_model: args.concurrency},
                                  limit=args.limit, client=cl)
        print(f"  执行       : 失败 {stats.get('failed')}")
        execute.write_manifest(
            run_id=args.out_run, configs=[],
            arms=[{"name": judge_model, "params": {"max_tokens": 512,
                                                   "temperature": 0.1,
                                                   "reasoning_effort": "none"}}],
            data_hash=_sha16(paths.data() / "eval_set.jsonl"),
            models_snapshot=[judge_model],
            note=f"badcase 归因第二标注者｜源 run={args.run_id}｜"
                 f"{len(specs)} 次（{args.reps} 遍）｜judge={judge_model}",
            stats=stats)

    # ---- 汇总 -------------------------------------------------------------
    judged = execute.load_results(args.out_run, Config.C4)
    by_key: dict[tuple[str, str, int], str] = {}
    for r in judged:
        if r.target_arm:
            by_key[(r.target_arm, r.instance_id, r.rep)] = parse_verdict(r.content)

    print()
    print("=" * 78)
    print("① LLM 自一致率（**噪声地板**：没有它，κ 无法解读）")
    print("=" * 78)
    self_pairs = []
    for (arm, iid, rep), v in by_key.items():
        if rep != 0:
            continue
        v1 = by_key.get((arm, iid, 1))
        if v1:
            self_pairs.append((v, v1))
    sa, sk = kappa(self_pairs)
    print(f"  {len(self_pairs)} 条跑了 2 遍：原始一致率 {sa:.1%}　κ = {sk:.3f}")
    if sa < 0.7:
        print(f"  ⚠️ 自一致率偏低 ⇒ **任务本身噪声大**，人机 κ 低不能全归咎于标签集")

    # ---- 人机一致率 -------------------------------------------------------
    human = load_human(args.run_id)
    print()
    print("=" * 78)
    print("② 人机一致率")
    print("=" * 78)
    pairs = []
    for (iid, arm), hv in human.items():
        lv = by_key.get((arm, iid, 0))
        if lv:
            pairs.append((hv, lv))
    if pairs:
        ag, kp = kappa(pairs)
        print(f"  共同标注 {len(pairs)} 条：原始一致率 {ag:.1%}　**Cohen's κ = {kp:.3f}**")
        print(f"  （LLM 自一致率 {sa:.1%} 是 κ 的上限参考）")
        if kp >= 0.6:
            print(f"  ✅ κ ≥ 0.6 ⇒ 归因分布可用")
        elif kp >= 0.4:
            print(f"  ⚠️ 0.4 ≤ κ < 0.6 ⇒ 只报「哪些类难标」，不报精确分布")
        else:
            print(f"  ❌ κ < 0.4 ⇒ 标签集需重做")
        print()
        print("  混淆矩阵（行=人工，列=LLM）：")
        cm = defaultdict(Counter)
        for h, l in pairs:
            cm[h][l] += 1
        print(f"    {'人工\\LLM':<20}" + "".join(f"{x[:9]:>11}" for x in ATTR_LABELS))
        for h in ATTR_LABELS:
            if not cm[h]:
                continue
            print(f"    {h:<20}" + "".join(f"{cm[h].get(x, 0):>11}" for x in ATTR_LABELS))
    else:
        print("  还没有人工标签可对比")

    # ---- 全量分布 + 按模型（同家族偏差检查）--------------------------------
    allv = [v for (_, _, rep), v in by_key.items() if rep == 0 and v != "UNCLEAR"]
    print()
    print("=" * 78)
    print(f"③ LLM 的归因分布（全部 {len(allv)} 条；未经人工确认，仅作预览）")
    print("=" * 78)
    for k, v in Counter(allv).most_common():
        print(f"  {k:<18}{v:>4}  {v/len(allv):>6.1%}")
    print()
    print("  按模型（★ judge 与 deepseek 同家族，必须按模型看）：")
    for arm in sorted({a for (a, _, rep) in by_key if rep == 0}):
        vs = [v for (a, _, rep), v in by_key.items() if a == arm and rep == 0]
        ge = sum(1 for v in vs if v == "gold_error")
        print(f"    {arm:<28} n={len(vs):<4} gold_error={ge}"
              f"({ge/len(vs):.0%})  {dict(Counter(vs).most_common(3))}")

    out = paths.ROOT / "runs" / args.out_run / "badcase_judge_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "judge_model": judge_model, "n_items": len(allv),
        "self_agreement": None if sa != sa else round(sa, 4),
        "self_kappa": None if sk != sk else round(sk, 3),
        "human_vs_llm_n": len(pairs),
        "distribution": dict(Counter(allv)),
        "skipped": [{"instance_id": a, "arm": b, "why": c} for a, b, c in skipped],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


def _sha16(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
