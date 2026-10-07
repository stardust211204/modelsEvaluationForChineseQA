"""把人工填好的标签表并回复核清单，并**先报一致率再报分布**。

纪律（见 docs/badcase_method.md）
    单标注者、无一致率证据的标签，与启发式归因**没有区别**。
    所以本脚本的输出顺序是固定的：
        ① 校验标签是否合法
        ② 报一致率（两个标注者/两轮之间）—— κ 与原始一致率都给
        ③ **只有**一致率过门槛才打印分布
    一致率不足时，分布一律不打印（避免被截图引用）。

用法
    # 单人一轮
    .venv\\Scripts\\python.exe scripts\\merge_badcase_labels.py --run-id c2-full

    # 双标注（两个 CSV 各带 annotator 列）
    .venv\\Scripts\\python.exe scripts\\merge_badcase_labels.py --run-id c2-full \\
        --labels runs/c2-full/badcase_labels_A.csv \\
        --labels runs/c2-full/badcase_labels_B.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import paths  # noqa: E402

LABELS = {
    "knowledge_gap", "reasoning_error", "option_confusion", "question_misread",
    "detail_omission", "gold_error", "format_parse", "cannot_tell",
}

KAPPA_USE = 0.60        # ≥0.6 分布可用
KAPPA_WEAK = 0.40       # 0.4–0.6 只报"哪些类难标"


def read_csv(p: Path) -> dict[tuple[str, str], str]:
    """读标签表。**按 (instance_id, arm) 匹配**，不依赖 id 列 ——
    人工表的列结构可能各不相同（实测 `reports/badcase_labels.csv` 就没有 id 列，
    还带若干 Excel 留下的空列，以及第一行右侧被塞进来的标签速查表）。
    """
    out: dict[tuple[str, str], str] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("confirmed_attribution") or "").strip()
            iid = (row.get("instance_id") or "").strip()
            arm = (row.get("arm") or "").strip()
            if not v or not iid or not arm:
                continue
            if v not in LABELS:
                # 不在这里报错：先收集，统一在 main 里给出可读的失败信息
                out[(iid, arm)] = v
                continue
            out[(iid, arm)] = v
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """返回 (原始一致率, κ)。κ 扣除"碰巧一致"的成分。"""
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    agree = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    return agree, kappa


def main() -> int:
    ap = argparse.ArgumentParser(description="合并 badcase 人工标签")
    ap.add_argument("--run-id", default="c2-full")
    ap.add_argument("--labels", action="append", default=None,
                    help="标签 CSV（可给多个 = 多标注者）。默认找 reports/ 与 runs/ 下的标签表")
    args = ap.parse_args()

    d = paths.ROOT / "runs" / args.run_id
    src = d / "badcase_review.jsonl"
    if not src.exists():
        print(f"!! 未找到 {src}")
        return 2
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ★ 优先用 reports/ 下人工维护的那份（合并计划里的唯一权威路径），
    #   找不到再回退到 runs/ 下的自动生成模板。
    if args.labels:
        files = [Path(x) for x in args.labels]
    else:
        files = [f for f in sorted((paths.ROOT / "reports").glob("*labels*.csv"))]
        if not files:
            files = sorted(d.glob("*labels*.csv"))
    files = [f for f in files if f.exists()]
    if not files:
        print("!! 没找到标签表。先跑 scripts/make_badcase_sheet.py 生成模板")
        return 2

    sheets = {f.stem: read_csv(f) for f in files}
    print("=" * 78)
    print(f"合并 badcase 人工标签   run={args.run_id}")
    print("=" * 78)
    for name, m in sheets.items():
        print(f"  {name:<42} 已填 {len(m)} 条")

    # ---- ① 校验 -----------------------------------------------------------
    bad = [(k, v) for m in sheets.values() for k, v in m.items() if v not in LABELS]
    if bad:
        print(f"\n  ❌ 非法标签 {len(bad)} 个（只能是那 8 个）：")
        for k, v in bad[:8]:
            print(f"     {k} -> {v!r}")
        return 3
    print(f"  ✅ 标签全部合法")

    # ---- ② 一致率（先报，再报分布）----------------------------------------
    names = list(sheets)
    kappa = None
    if len(names) >= 2:
        keys = set(sheets[names[0]]) & set(sheets[names[1]])
        pairs = [(sheets[names[0]][k], sheets[names[1]][k]) for k in keys]
        agree, kappa = cohen_kappa(pairs)
        print()
        print(f"  双标注一致率（{names[0]} vs {names[1]}，共同标了 {len(pairs)} 条）：")
        print(f"    原始一致率 = {agree:.1%}    Cohen's κ = {kappa:.3f}")
        if kappa < KAPPA_WEAK:
            print(f"    ❌ κ < {KAPPA_WEAK} ⇒ **标签集本身要重做**，分布不打印")
        elif kappa < KAPPA_USE:
            print(f"    ⚠️ {KAPPA_WEAK} ≤ κ < {KAPPA_USE} ⇒ 只报「哪些类难标」，**不报精确分布**")
        else:
            print(f"    ✅ κ ≥ {KAPPA_USE} ⇒ 分布可用")
        # 逐类一致率（哪几类难标）
        print(f"\n    逐类一致率：")
        for lbl in sorted(LABELS):
            sub = [(a, b) for a, b in pairs if a == lbl or b == lbl]
            if len(sub) < 3:
                continue
            ag = sum(1 for a, b in sub if a == b) / len(sub)
            print(f"      {lbl:<18} n={len(sub):<4} 一致 {ag:.0%}")
    else:
        print(f"\n  ⚠️ 只有 {len(names)} 份标签表 —— **单标注者无一致率证据**。")
        print(f"     按 docs/badcase_method.md，此时**不得报精确分布**。")
        print(f"     要么再标一轮（同一人隔 ≥1 天），要么让我用 LLM 作第二标注者。")

    # ---- 合并落盘（保留原文件）--------------------------------------------
    merged = []
    for r in rows:
        k = (r["instance_id"], r["arm"])
        vals = [m[k] for m in sheets.values() if k in m]
        r = dict(r)
        r["confirmed_attribution"] = vals[0] if vals else None
        r["annotator"] = "+".join(n for n, m in sheets.items() if k in m) or ""
        merged.append(r)
    outp = d / "badcase_review_labeled.jsonl"
    outp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                    encoding="utf-8")
    n_done = sum(1 for r in merged if r["confirmed_attribution"])
    print(f"\n  已标注 {n_done} / {len(merged)} -> {outp.relative_to(paths.ROOT)}")

    # ---- ③ 分布（仅在一致率足够时）----------------------------------------
    if kappa is not None and kappa >= KAPPA_USE and n_done:
        c = Counter(r["confirmed_attribution"] for r in merged if r["confirmed_attribution"])
        print()
        print("=" * 78)
        print("归因分布（κ 已过门槛）")
        print("=" * 78)
        for k, v in c.most_common():
            print(f"  {k:<18}{v:>4}  {v/n_done:>6.1%}")
        ge = c.get("gold_error", 0)
        if ge:
            print()
            print(f"  ⚠️ **{ge} 条判为 gold_error** —— 若成立，已发布的准确率数字需要重算，")
            print(f"     并在报告中说明修正幅度。这些条目：")
            for r in merged:
                if r["confirmed_attribution"] == "gold_error":
                    print(f"       {r['instance_id']} ({r['arm']})  {r.get('reviewer_note','')[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
