"""构建 E2b「打乱标签但保持金标字母」题集 —— E2 的对照。

为什么需要它
    E2 的题面是 `(C) … (A) … (D) … (B) …` 这种**乱序标签**排版，
    **真实考试里不存在**。所以 E2 相对 C1 的准确率下降（−4.5 ~ −18.8pp）
    有两个可能来源，E2 单独**分不开**：

        (a) 字母↔内容的对应被破坏   —— 模型依赖"答 A 就是第一个选项"这类表面线索
        (b) 题面排版异常本身         —— 模型没见过这种排版，能力下降

本实验把 (b) 单独隔离出来：

    C1   标签顺序排列，内容原序          基准
    E2   标签打乱，内容原序              (a) + (b) 都在
    E2b  标签打乱，**但金标槽位仍显示它原本的字母**   ← 只有 (b)

    ★ 为什么 E2b 能隔离 (b)
      在 C1 里，金标内容在槽位 g、显示字母 LETTERS[g]。
      在 E2b 里，**同样是槽位 g、同样是字母 LETTERS[g]** —— 只是其余选项的
      标签被打乱了。于是：

        · 「偏好字母 A」的模型      → 两个条件下命中率相同
        · 「偏好第 k 个槽位」的模型 → 两个条件下命中率相同

      唯一变化的是**题面看起来乱不乱**。

    于是分解为：
        C1 − E2b  = 排版异常效应
        E2b − E2  = 字母↔内容对应被破坏的效应

不干扰已有数据
    只新增 `data/e2b_relabel_set.jsonl` 与 `data/e2b_relabel_set.meta.json`；
    不改 eval_set / matrix.yaml / exclusions.json / src/meval，
    复用已有的 `prompts/e2_labelshuffle.txt` 模板。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, paths  # noqa: E402
from meval.types import Instance, Source, Split  # noqa: E402

SEED = 20261004
SRC_SET = "eval_set.jsonl"
SRC_RUN = ("c6-5arm-v2", "c6")     # ★ 与 E2 / C6 / C3 同一批 100 题，保证可配对
LETTERS = "ABCDEFGHIJ"


def label_order_gold_fixed(inst_id: str, n: int, gold: int) -> list[int]:
    """确定性打乱标签，**但金标槽位仍显示它原本的字母**。

    `L[i]` = 第 i 个槽位显示第几个字母。
    约束：`L[gold] == gold` ⇒ 槽位 gold 显示 `LETTERS[gold]`，与 C1 一致。
    其余槽位与其余字母之间做随机双射（仍是排列）。
    """
    rng = random.Random(f"{SEED}:labelb:{inst_id}")
    others = [c for c in range(n) if c != gold]
    rng.shuffle(others)
    L = [0] * n
    L[gold] = gold
    for slot, letter in zip((i for i in range(n) if i != gold), others):
        L[slot] = letter
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 E2b 对照题集")
    ap.add_argument("--out", default="e2b_relabel_set.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    src = paths.data() / SRC_SET
    pool = {i.id: i for i in build_set.load_frozen(src)}

    want: list[str] = []
    seen: set[str] = set()
    rid, cfg = SRC_RUN
    for fp in (paths.ROOT / "runs" / rid / "raw" / cfg).glob("*.jsonl"):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            iid = json.loads(ln)["instance_id"]
            if iid not in seen:
                seen.add(iid)
                want.append(iid)
    want.sort()
    print(f"  题源 : {SRC_SET}    取自 {rid} 的 {len(want)} 题（与 E2 同一批）")

    out: list[Instance] = []
    meta: dict[str, list[int]] = {}
    for iid in want:
        inst = pool[iid]
        n = len(inst.options)
        gold = inst.gold_indices[0]
        L = label_order_gold_fixed(iid, n, gold)
        assert sorted(L) == list(range(n)), f"{iid}: L 不是排列"
        assert L[gold] == gold, f"{iid}: 金标字母未被保持"
        meta[iid] = L

        block = "\n".join(
            f"({LETTERS[L[i]]}) {inst.options[i]}" for i in range(n))
        new_q = f"{inst.question}\n\n{block}"

        out.append(Instance(
            id=inst.id, question=new_q,
            options=list(inst.options),            # ★ 原顺序，判分逻辑不动
            gold_indices=list(inst.gold_indices),  # ★ 金标不动
            split=Split.TEST if inst.split is Split.TEST else inst.split,
            stratum={**dict(inst.stratum), "label_order": L},
            source=inst.source if hasattr(inst, "source") else Source.PUBLIC,
            note="E2b 对照：标签打乱但金标字母保持；分析期用 label_order.index(ans)",
        ))

    # ---- 自检 --------------------------------------------------------------
    print("\n  自检：")
    bad = []
    for a, b in zip(out, (pool[i] for i in want)):
        if a.options != b.options:
            bad.append(f"{a.id}: 选项被改动")
        if a.gold_indices != b.gold_indices:
            bad.append(f"{a.id}: 金标被改动")
    print(f"    选项/金标 一致：{'✅' if not bad else '★ ' + str(bad[:3])}")

    # ★ 核心约束：金标字母必须与 C1 一致
    same = sum(1 for iid in want if meta[iid][pool[iid].gold_indices[0]]
               == pool[iid].gold_indices[0])
    print(f"    ★ 金标字母与 C1 一致：{same}/{len(want)} "
          f"{'✅' if same == len(want) else '★'}")

    # 排版确实被打乱了（非金标槽位里，标签与槽位下标相等的比例应远低于 1）
    scrambled = sum(1 for iid in want
                    for s, l in enumerate(meta[iid])
                    if s != pool[iid].gold_indices[0] and s == l)
    total_other = sum(len(meta[iid]) - 1 for iid in want)
    print(f"    非金标槽位中「标签==槽位」的比例 "
          f"{scrambled}/{total_other} = {scrambled/total_other:.3f}"
          f"（随机应约 {1/9:.3f}；若接近 1 说明没打乱）")

    # 与 E2 的映射做对照：E2 里金标字母是随机的
    e2meta = paths.data() / "e2_relabel_set.meta.json"
    if e2meta.exists():
        e2L = json.loads(e2meta.read_text(encoding="utf-8"))["label_order"]
        e2_same = sum(1 for iid in want
                      if e2L.get(iid, [None])[pool[iid].gold_indices[0]]
                      == pool[iid].gold_indices[0])
        print(f"    （对照）E2 里金标字母与 C1 一致的题数：{e2_same}/{len(want)}"
              f"  ← 应明显少于 {len(want)}")

    print("\n  样例（前 2 题）：")
    for inst in out[:2]:
        print(f"\n  ── {inst.id}  L={meta[inst.id]}  gold={inst.gold_indices}")
        for ln in inst.question.splitlines()[-4:]:
            print(f"      {ln[:110]}")

    if args.dry_run:
        print("\n  --dry-run：未写文件")
        return 0

    dest = paths.data() / args.out
    h = build_set.freeze(out, dest)
    (paths.data() / "e2b_relabel_set.meta.json").write_text(
        json.dumps({"source": SRC_SET, "source_run": rid, "seed": SEED,
                    "label_order": meta,
                    "note": "E2b 对照：标签打乱但金标字母保持；分析期用 "
                            "content_index = label_order.index(answer_index)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ 题集 -> data/{args.out}    hash={h}")
    print(f"  ✅ 映射 -> data/e2b_relabel_set.meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
