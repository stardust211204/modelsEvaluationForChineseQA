"""构建 E2「选项标签打乱」题集。

目的：分离「末位槽位偏好」与「选项 ID token 偏好」。

设计
    C6 的乱序只移动**内容**，标签 A/B/C/D 始终钉在槽位上
    ⇒ 「偏好字母 A」与「偏好第 1 个槽位」在该设计下**不可区分**。
    本实验改为：内容与槽位都不动，只打乱**标签到槽位的映射**。

        slot:   0     1     2     3
        label:  C     A     D     B        ← L = [2, 0, 3, 1]

    于是两个假说给出**相反**的信号：

        假说                        错误的槽位分布        错误的字母分布
        末位槽位偏好（slot 3）      集中在 slot 3         均匀
        选项 ID token 偏好（D）     均匀                  集中在字母 D

★ 为什么不用「换字母表」（如 ①②③④）
    换字母表会让模板里的「A 到 {last}」与实际标签矛盾，且解析器只认 A–J。
    打乱映射保留了 A–J 字母表 ⇒ **模板与解析器都不用改**，只需在分析期把
    「模型答的字母」映射回「内容下标」：

        content_index = label_order.index(tally.answer_index)

    （注意方向与 C6 的 `unpermute_answer` **相反**：C6 是 perm[ans]，本实验是
      label_order.index(ans)。）

不干扰已有数据
    只新增 `data/e2_relabel_set.jsonl` 与 `data/e2_relabel_set.meta.json`；
    不改 eval_set / matrix.yaml / exclusions.json / src/meval。
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
SRC_RUN = ("c6-5arm-v2", "c6")     # ★ 沿用 C6 那 100 题，保证与 C6/C3 配对
LETTERS = "ABCDEFGHIJ"


def label_order(inst_id: str, n: int) -> list[int]:
    """确定性打乱标签：`L[i]` = 第 i 个槽位显示第几个字母。

    种子只依赖 instance_id ⇒ 同一题永远同一映射（可复现、可配对）。
    """
    rng = random.Random(f"{SEED}:label:{inst_id}")
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 E2 选项标签打乱题集")
    ap.add_argument("--out", default="e2_relabel_set.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    src = paths.data() / SRC_SET
    pool = {i.id: i for i in build_set.load_frozen(src)}

    # ---- 取出 C6 那 100 题 -------------------------------------------------
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
    print(f"  题源 : {SRC_SET}    取自 {rid} 的 {len(want)} 题")

    out: list[Instance] = []
    meta: dict[str, list[int]] = {}
    for iid in want:
        inst = pool[iid]
        n = len(inst.options)
        L = label_order(iid, n)
        assert sorted(L) == list(range(n)), f"{iid}: L 不是排列"
        meta[iid] = L

        # 把带标签的选项块**烤进题干**（模板不再使用 {options}）
        block = "\n".join(
            f"({LETTERS[L[i]]}) {inst.options[i]}" for i in range(n))
        new_q = f"{inst.question}\n\n{block}"

        out.append(Instance(
            id=inst.id, question=new_q,
            options=list(inst.options),          # ★ 保持原顺序，判分逻辑不动
            gold_indices=list(inst.gold_indices),  # ★ 金标不动
            split=Split.TEST if inst.split is Split.TEST else inst.split,
            stratum={**dict(inst.stratum), "label_order": L},
            source=inst.source if hasattr(inst, "source") else Source.PUBLIC,
            note="E2 标签打乱：字母→槽位映射已打乱，分析期需 label_order.index(ans)",
        ))

    # ---- 自检 --------------------------------------------------------------
    print("\n  自检：")
    bad = []
    for a, b in zip(out, (pool[i] for i in want)):
        if a.options != b.options:
            bad.append(f"{a.id}: 选项被改动")
        if a.gold_indices != b.gold_indices:
            bad.append(f"{a.id}: 金标被改动")
        if len(a.options) != len(b.options):
            bad.append(f"{a.id}: 选项数变了")
        if str(a.gold_indices[0]) not in "0123456789":
            bad.append(f"{a.id}: 金标非法")
    print(f"    选项/金标/选项数 一致：{'✅' if not bad else '★ ' + str(bad[:3])}")
    dist = {}
    for L in meta.values():
        dist[len(L)] = dist.get(len(L), 0) + 1
    print(f"    选项数分布 {dict(sorted(dist.items()))}")
    # 打乱是否真的生效：首槽位显示 A 的比例应接近 1/n
    first = sum(1 for L in meta.values() if L[0] == 0) / len(meta)
    print(f"    首槽位恰好显示 A 的比例 {first:.3f}（随机应约 0.10–0.25）")

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
    (paths.data() / "e2_relabel_set.meta.json").write_text(
        json.dumps({"source": SRC_SET, "source_run": rid, "seed": SEED,
                    "label_order": meta, "note": "E2 标签打乱映射；分析期用 "
                    "content_index = label_order.index(answer_index)"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ 题集 -> data/{args.out}    hash={h}")
    print(f"  ✅ 映射 -> data/e2_relabel_set.meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
