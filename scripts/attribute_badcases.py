"""badcase 归因：把错题按可能原因分档，并产出**人工复核清单**。

铁律：**启发式只提议，结论必须人工背书。**
    本项目 Limitations 已声明归因是启发式。所以这个脚本的主产物不是"归因分布"，
    而是 `badcase_review.jsonl` —— 一份带完整上下文、`confirmed` 字段留空的复核清单。
    没有人工回填的分布**不得写进报告**。

★ 为什么默认跑 C2 而不是 C1
    C1 的强格式约束把作答压成单个字母（content 中位长度 **1** 字符），
    推理完全不可见 —— 归因只能靠"选项是否相邻"这一条，几乎退化成瞎猜。
    C2 的自由作答把推理写进了 `content`（中位 **536** 字符），才谈得上归因。
    **这本身是一条值得写进报告的结论：格式约束的代价不止是准确率，
    还包括"事后无法诊断"。**

用法
    .venv\\Scripts\\python.exe scripts\\attribute_badcases.py                    # 默认 c2-full
    .venv\\Scripts\\python.exe scripts\\attribute_badcases.py --run-id c2-full --n-review 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode, Tier  # noqa: E402

SEED = 20261004

# 人工标签集（闭合、可判定）。见 docs/badcase_method.md
LABELS = [
    "knowledge_gap",     # 知识缺失
    "reasoning_error",   # 推理链错误
    "option_confusion",  # 选项混淆
    "question_misread",  # 误读题干
    "detail_omission",   # 遗漏关键细节
    "gold_error",        # ★ 金标准有误 —— ErrorMap 把 dataset noise 与模型能力并列
    "format_parse",      # 格式/解析（我们自己的口径问题）
    "cannot_tell",       # 无法判断
]


def main() -> int:
    ap = argparse.ArgumentParser(description="badcase 归因")
    ap.add_argument("--run-id", default="c2-full")
    ap.add_argument("--config", default="c2")
    ap.add_argument("--mode", default="lenient", choices=["strict", "lenient"],
                    help="C2 的严格口径全不可判，故默认宽松")
    ap.add_argument("--n-review", type=int, default=120, help="人工复核清单条数")
    ap.add_argument("--write", action="store_true", default=True)
    args = ap.parse_args()

    cfg = Config(args.config)
    mode = ParseMode(args.mode)
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    res = {(r.arm, r.instance_id): r for r in execute.load_results(args.run_id, cfg)}
    rows = [t for t in metrics.load_tallies(args.run_id, cfg) if t["mode"] == mode.value]
    if not rows:
        print(f"!! runs/{args.run_id}/tallies/{cfg.value}/all.jsonl 为空")
        return 2

    arms = sorted({t["arm"] for t in rows})
    print("=" * 78)
    print(f"badcase 归因   run={args.run_id}  口径={mode.value}")
    print("=" * 78)

    # ---- 逐条归因 ----------------------------------------------------------
    cases: list[dict] = []
    for t in rows:
        if t["tier"] != Tier.WRONG.value:
            continue
        r = res.get((t["arm"], t["instance_id"]))
        inst = pool.get(t["instance_id"])
        if r is None or inst is None:
            continue
        attr = tally.guess_attribution(r, inst, _as_tally(t))
        content = r.content or ""
        gold_texts = [inst.options[g] for g in inst.gold_indices
                      if g < len(inst.options) and len(inst.options[g]) >= 4]
        cases.append({
            "instance_id": t["instance_id"],
            "arm": t["arm"],
            "subject": (inst.stratum or {}).get("subject", "unknown"),
            "gold_indices": inst.gold_indices,
            "gold_letters": inst.gold_letters,
            "model_answer_index": t.get("answer_index"),
            "tier_step": t.get("tier_step"),
            "heuristic_attribution": attr.value if attr else None,
            # 给人工复核用的额外信号：是否**提到过**正确选项的文本（"擦肩而过"）
            "mentions_gold_text": any(g in content for g in gold_texts) if gold_texts else None,
            "adjacent_to_gold": (
                t.get("answer_index") is not None
                and any(abs(t["answer_index"] - g) == 1 for g in inst.gold_indices)),
            "content_len": len(content),
            # ★ 留空：必须人工回填后分布才可用
            "confirmed_attribution": None,
            "reviewer_note": "",
            "label_options": LABELS,
            "annotator": "",      # 双标注：填 A（人）或 B（LLM）
            "round": 1,           # 重测一致性：同一标注者第几轮
        })

    n_wrong = len(cases)
    if not n_wrong:
        print("  没有 wrong 档条目（该口径下可能全不可判）")
        return 0

    # ---- 分布 --------------------------------------------------------------
    print(f"  错题总数 : {n_wrong}（{mode.value} 口径）")
    print(f"  可归因素材: content 中位长度 "
          f"{sorted(c['content_len'] for c in cases)[n_wrong // 2]} 字符\n")

    print(f"  {'臂':<28}{'错题':>6}  归因分布（启发式，**未人工确认**）")
    by_arm: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        by_arm[c["arm"]][c["heuristic_attribution"]] += 1
    for arm in arms:
        cc = by_arm.get(arm)
        if not cc:
            continue
        tot = sum(cc.values())
        dist = "  ".join(f"{k}={v}({v/tot:.0%})" for k, v in cc.most_common())
        print(f"  {arm:<28}{tot:>6}  {dist}")

    print()
    print(f"  {'学科':<20}{'错题':>6}{'擦肩而过(提到正确项)':>22}")
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_subj[c["subject"]].append(c)
    for subj, cs in sorted(by_subj.items(), key=lambda kv: -len(kv[1]))[:8]:
        near = sum(1 for c in cs if c["mentions_gold_text"])
        print(f"  {subj:<20}{len(cs):>6}{near:>22}")

    # ---- 诚实交代启发式有多粗 ---------------------------------------------
    # `reasoning` 档的判据是"推理文本 ≥ 400 字符"。在 C2 里作答中位长度就有 677 字符，
    # 所以这一档**实际上主要在区分"话多话少"**，不是在区分"推理错 vs 不知道"。
    # 不把这一点说清楚，那张分布表会被当成结论引用。
    by_attr_len: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        by_attr_len[c["heuristic_attribution"]].append(c["content_len"])
    print()
    print("  ⚠️ 启发式有多粗（必须连同分布一起引用）：")
    for k, v in sorted(by_attr_len.items()):
        v = sorted(v)
        print(f"     {k:<18} n={len(v):<4} content 长度 中位 {v[len(v)//2]:<5} "
              f"最小 {v[0]:<5} 最大 {v[-1]}")
    print("     `reasoning` 档的判据就是「推理文本 ≥ 400 字符」。C2 作答中位 677 字符，")
    print("     所以这一档**主要在区分话多话少**，不是在区分「推理错」与「不知道」。")
    print("     ⇒ 这张表只能用来**挑出该看哪些题**，不能当结论。")

    # ---- 人工复核清单（主产物）--------------------------------------------
    # ★ 分层维度必须是**独立于归因**的，否则就是自我循环。
    #   第一版按 (arm × 启发式归因) 分层 —— 而该启发式实测只是"文本长度带"，
    #   于是抽样也偏向了"话多"的题，等于用待验证的标签去决定验证谁。
    #   改为 (arm × 学科) + 强制纳入最有信息量的两类。
    rng = random.Random(SEED)

    forced: list[dict] = []
    forced_ids: set[tuple[str, str]] = set()

    def _force(pred, tag: str) -> None:
        for c in cases:
            k = (c["arm"], c["instance_id"])
            if k in forced_ids or not pred(c):
                continue
            forced_ids.add(k)
            c = dict(c, forced_reason=tag)
            forced.append(c)

    # ① "擦肩而过"：推理里提到过正确项却仍选错 —— 信息量最高
    _force(lambda c: c["mentions_gold_text"] is True, "mentions_gold_text")
    # ② 相邻选项：验证"相邻=混淆"这个假设到底成不成立
    _force(lambda c: c["adjacent_to_gold"] is True, "adjacent_to_gold")

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for c in cases:
        if (c["arm"], c["instance_id"]) in forced_ids:
            continue
        buckets[(c["arm"], c["subject"])].append(c)
    for v in buckets.values():
        rng.shuffle(v)

    target = min(args.n_review, n_wrong)
    review: list[dict] = forced[:target]
    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    i = 0
    while len(review) < target:
        added = False
        for k in keys:
            if i < len(buckets[k]) and len(review) < target:
                review.append(buckets[k][i])
                added = True
        if not added:
            break
        i += 1
    rng.shuffle(review)

    rp = paths.ROOT / "runs" / args.run_id / "badcase_review.jsonl"
    rp.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in review) + "\n",
                  encoding="utf-8")

    summary = {
        "run_id": args.run_id, "config": cfg.value, "mode": mode.value,
        "n_wrong": n_wrong,
        "by_arm": {a: dict(c) for a, c in by_arm.items()},
        "by_subject": {s: len(cs) for s, cs in by_subj.items()},
        "n_review": len(review),
        "review_stratified_by": ["arm", "subject"],
        "forced_included": dict(Counter(c.get("forced_reason") for c in review
                                        if c.get("forced_reason"))),
        "label_options": LABELS,
        "confirmed": False,
        "note": "启发式归因**未经人工确认**，不得写进报告结论。"
                "复核清单见 badcase_review.jsonl（confirmed_attribution 待回填）。"
                "回填后需按 docs/badcase_method.md 步骤 3 先报一致率（κ/α）再报分布。",
    }
    sp = paths.ROOT / "runs" / args.run_id / "badcase_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("人工复核清单（主产物）")
    print("=" * 78)
    is_census = len(review) >= n_wrong
    print(f"  条数     : {len(review)}（目标 {args.n_review}；错题总数 {n_wrong}）")
    print(f"  分层方式 : arm × 学科（★ 独立于归因，避免用待验证的标签决定验证谁）")
    n_forced = sum(1 for c in review if c.get("forced_reason"))
    print(f"  强制纳入 : {n_forced} 条"
          f"（{summary['forced_included']}）—— 信息量最高的两类")
    if is_census:
        print(f"  ✅ **全覆盖**：清单已包含全部 {n_wrong} 条错题，是**普查不是抽样**，")
        print(f"     因此回填后**可以直接得到总体归因分布**，无需担心抽样偏差。")
    else:
        print(f"  ⚠️ 这是**分层抽样**，不是均匀随机样本，不能用它反推总体比率。")
    print(f"  标签集   : {len(LABELS)} 类 -> {', '.join(LABELS)}")
    print(f"  回填字段 : confirmed_attribution + reviewer_note + annotator + round")
    print(f"\n  清单 -> {rp.relative_to(paths.ROOT)}")
    print(f"  汇总 -> {sp.relative_to(paths.ROOT)}")
    print()
    print("  ★ 归因分布**尚未可用**。回填后必须**先报一致率（κ 或 α）再报分布** ——")
    print("    单标注者、无一致率证据的标签，与启发式归因没有区别。")
    print("    方法见 docs/badcase_method.md（含双标注协议与 κ 判定门槛）。")
    return 0


def _as_tally(d: dict):
    """把 tally dict 还原成 Tally 对象（guess_attribution 需要它）。"""
    from meval.types import Tally

    return Tally(
        instance_id=d["instance_id"], arm=d["arm"], config=Config(d["config"]),
        rep=d.get("rep", 0), mode=ParseMode(d["mode"]), tier=Tier(d["tier"]),
        answer_index=d.get("answer_index"), tier_step=d.get("tier_step", ""),
        invalid_reason=d.get("invalid_reason", ""),
        attribution=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
