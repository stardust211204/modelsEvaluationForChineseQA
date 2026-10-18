"""位置敏感性分析（C6）。

为什么 C6 要和 C3 用**同一批 100 题**
    "乱序让 30% 的题改了答案"这种数字，单独看**无法解读** ——
    模型自己重复 10 次也会改答案。所以必须有一个噪声地板。
    C6 与 C3 共用同一批 100 题，于是：

        乱序翻转率  vs  重复翻转率（噪声地板）

    两者同题可比。这才是"位置敏感性到底有多大"的可读答案。

三个关键量
    ① **选项翻转率**：把作答还原到**原题选项下标**后，3 个排列里答案是否变过。
       ★ 必须还原：模型每次都选 "B" 会被误读成"很稳定"，
         而 B 在不同排列里是**不同的选项**。
    ② **字母黏着率**：选中的**字母**是否始终相同（不管它对应哪个选项）。
       字母黏着率显著高于选项一致率 ⇒ 模型被"位置"吸引，而不是被内容驱动。
    ③ **敏感度极差**：各排列准确率的 max − min（与文献 sensitivity gap 同口径）。

用法
    .venv\\Scripts\\python.exe scripts\\analyze_position.py
    .venv\\Scripts\\python.exe scripts\\analyze_position.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, metrics, paths  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402

STRICT = ParseMode.STRICT.value


def load_pool() -> dict:
    return {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}


def orig_frame(tallies: list[dict], arm: str, pool: dict, *,
               permuted: bool) -> dict[tuple[str, int], int | None]:
    """(instance_id, rep) -> **原题选项下标**。

    ★ `permuted` 必须显式传：
      C6 的作答下标在乱序坐标系里，要用同一排列还原；
      **C3 没有乱序**，它的 rep 只是重复序号，作答本来就在原题坐标系里。
      对 C3 也去"还原"会用一个从未施加过的排列去搅乱它 ——
      表现为重复翻转率恒为 100%，而且看起来还挺像回事。
    """
    out: dict[tuple[str, int], int | None] = {}
    for t in tallies:
        if t["arm"] != arm or t["mode"] != STRICT:
            continue
        inst = pool.get(t["instance_id"])
        if inst is None:
            continue
        idx = t.get("answer_index")
        if permuted:
            _, perm = build_set.permute_instance(inst, t.get("rep", 0))
            idx = build_set.unpermute_answer(idx, perm)
        out[(t["instance_id"], t.get("rep", 0))] = idx
    return out


def pairwise_disagreement(fr: dict[tuple[str, int], int | None],
                          reps: list[int]) -> tuple[float, int]:
    """**逐对**不一致率：两次不同观测给出不同答案的概率。

    ★ 这是唯一能与不同抽样次数比较的口径。
      "在 k 次里是否变过"（翻转率）随 k 单调上升：
      C3 有 10 次、C6 只有 3 个排列，直接比翻转率会**系统性地偏向重复侧**。
      逐对率是每次配对的概率，与抽取次数无关。
    """
    eq = n = 0
    for iid in {k[0] for k in fr}:
        vals = [fr.get((iid, r)) for r in reps]
        clean = [v for v in vals if v is not None]
        for a in range(len(clean)):
            for b in range(a + 1, len(clean)):
                n += 1
                eq += int(clean[a] == clean[b])
    return (1 - eq / n if n else float("nan")), n


def flip_rate(fr: dict[tuple[str, int], int | None], reps: list[int]) -> tuple[float, int]:
    """在给定观测里"是否变过"。只用于同观测数的场合。"""
    n = ok = 0
    for iid in {k[0] for k in fr}:
        clean = [fr.get((iid, r)) for r in reps]
        clean = [v for v in clean if v is not None]
        if len(clean) < 2:
            continue
        n += 1
        ok += int(len(set(clean)) > 1)
    return (ok / n if n else float("nan")), n


def letter_frame(tallies: list[dict], arm: str) -> dict[tuple[str, int], int | None]:
    """(instance_id, rep) -> 模型**实际选的字母下标**（乱序坐标系）。"""
    return {(t["instance_id"], t.get("rep", 0)): t.get("answer_index")
            for t in tallies if t["arm"] == arm and t["mode"] == STRICT}


def main() -> int:
    ap = argparse.ArgumentParser(description="C6 位置敏感性分析")
    ap.add_argument("--c6-run", default="c6-full")
    ap.add_argument("--c3-run", default="c3-full", help="噪声地板来源（同一批 100 题）")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    pool = load_pool()
    c6 = metrics.load_tallies(args.c6_run, Config.C6)
    c3 = metrics.load_tallies(args.c3_run, Config.C3)
    if not c6:
        print(f"!! runs/{args.c6_run}/tallies/c6/all.jsonl 为空")
        return 2
    arms = sorted({t["arm"] for t in c6})

    print("=" * 84)
    print(f"位置敏感性（C6）   run={args.c6_run}   噪声地板来源={args.c3_run}")
    print("=" * 84)
    print("  排列 0 = 原顺序（内部基线）；排列 1/2/3 = 三次乱序。")
    print("  所有作答已还原到**原题选项下标**后再比较。\n")

    detail: dict = {"c6_run": args.c6_run, "c3_run": args.c3_run, "arms": {}}

    print(f"  {'模型':<28}{'各排列准确率':>28}{'极差pp':>9}{'排列0准确率':>12}")
    for arm in arms:
        rows = [t for t in c6 if t["arm"] == arm and t["mode"] == STRICT]
        by_rep = defaultdict(list)
        for t in rows:
            by_rep[t.get("rep", 0)].append(t["tier"])
        accs = {k: metrics.tier_accounting(v)["acc"] for k, v in sorted(by_rep.items())}
        vals = [a for a in accs.values() if a is not None]
        gap = round((max(vals) - min(vals)) * 100, 2) if vals else None
        shown = " ".join(f"k{k}={a:.1%}" if a is not None else f"k{k}=—"
                         for k, a in accs.items())
        print(f"  {arm:<28}{shown:>28}{gap if gap is not None else '—':>9}"
              f"{(accs.get(0) or 0):>12.1%}")
        detail["arms"][arm] = {"acc_by_perm": {str(k): v for k, v in accs.items()},
                               "sensitivity_gap_pp": gap}

    print()
    print("-" * 84)
    print("★ 核心对比：乱序 vs 重复（同一批 100 题，故可直接比）")
    print("-" * 84)
    print("  主口径 = **逐对不一致率**（与抽取次数无关）；副口径 = 翻转率（同为 3 次观测）")
    print(f"  {'模型':<28}{'乱序逐对':>10}{'重复逐对':>10}{'倍数':>8}"
          f"{'乱序翻转':>10}{'重复翻转':>10}{'字母黏着':>10}{'选项一致':>10}")
    for arm in arms:
        o = orig_frame(c6, arm, pool, permuted=True)
        lf = letter_frame(c6, arm)
        c3o = orig_frame(c3, arm, pool, permuted=False)

        # 主口径：逐对不一致率（C6 用排列 1/2/3；C3 用全部 10 次）
        order_pd, n_ord = pairwise_disagreement(o, [1, 2, 3])
        rep_pd, n_rep = pairwise_disagreement(c3o, list(range(10)))
        ratio = (order_pd / rep_pd) if rep_pd else float("nan")

        # 副口径：翻转率，两侧都限制为 3 次观测，避免"次数越多越容易翻转"的偏差
        order_flip, _ = flip_rate(o, [1, 2, 3])
        rep_flip3, _ = flip_rate(c3o, [0, 1, 2])
        letter_flip, _ = flip_rate(lf, [1, 2, 3])

        print(f"  {arm:<28}{order_pd:>10.1%}{rep_pd:>10.1%}{ratio:>8.2f}"
              f"{order_flip:>10.1%}{rep_flip3:>10.1%}"
              f"{1 - letter_flip:>10.1%}{1 - order_flip:>10.1%}")
        detail["arms"][arm].update({
            "order_pairwise_disagree": round(order_pd, 4),
            "repeat_pairwise_disagree": round(rep_pd, 4),
            "order_over_repeat_pairwise": round(ratio, 3) if ratio == ratio else None,
            "order_flip_rate_3": round(order_flip, 4),
            "repeat_flip_rate_3": round(rep_flip3, 4),
            "letter_stick_rate": round(1 - letter_flip, 4),
            "option_same_rate": round(1 - order_flip, 4),
            "n_order_pairs": n_ord, "n_repeat_pairs": n_rep,
        })

    print()
    print("读法")
    print("  * **逐对不一致率的倍数**是核心数字：")
    print("    ≈1 表示乱序的影响与模型自身抖动同量级（位置不敏感）；")
    print("    显著 >1 表示乱序真的在改答案，且改得比模型自己重复还多。")
    print("  * 翻转率那一组两侧都只用 3 次观测 —— 用 10 次比 3 次会系统性偏向重复侧。")
    print("  * **字母黏着率** 高于 **选项一致率** ⇒ 模型被位置吸引（记住了'选 B'），")
    print("    而 B 在不同排列里是不同选项。两者越接近，说明作答越由内容驱动。")
    print("  * **极差** 是各排列准确率的 max−min，与文献 sensitivity gap 同口径；")
    print("    但只有 4 个排列，它天然小于文献用 oracle 排序得到的上界，**不可直接对标**。")

    if args.write:
        out = paths.ROOT / "runs" / args.c6_run / "position_analysis.json"
        out.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
