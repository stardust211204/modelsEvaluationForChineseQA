"""生成 `configs/exclusions.json`：把**已确认有质量缺陷**的题列出来。

依据与设计动机见 `src/meval/exclusions.py` 的模块注释。一句话：
**题集不动，只记一份"这些题不可用于统计"的清单**，从而既提高口径质量，
又不破坏任何历史 run 的可比性（改题集会改哈希 ⇒ 门禁全 FAIL）。

用法
    .venv\\Scripts\\python.exe scripts\\build_exclusions.py            # 生成
    .venv\\Scripts\\python.exe scripts\\build_exclusions.py --dry-run  # 只看会剔哪些
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, exclusions, paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成题集排除清单")
    ap.add_argument("--dataset", default="eval_set")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    pool = build_set.load_frozen(paths.data() / f"{ds}.jsonl")
    print("=" * 84)
    print(f"生成排除清单  题集={ds}  共 {len(pool)} 题")
    print("=" * 84)

    hits: dict[str, list[str]] = {}
    for inst in pool:
        reasons = exclusions.detect_instance_defects(inst)
        if reasons:
            hits[inst.id] = reasons

    print(f"  命中规则的题：**{len(hits)}** / {len(pool)} = {len(hits)/len(pool):.1%}")
    structural = {desc for _, desc in exclusions.STRUCTURAL_RULES}
    n_struct = sum(1 for rs in hits.values() if any(r in structural for r in rs))
    print(f"    其中含结构化判据（选项重复）：{n_struct} 题")
    byreason = Counter(r for rs in hits.values() for r in rs)
    structural = {desc for _, desc in exclusions.STRUCTURAL_RULES}
    print("    文本正则：")
    for desc, n in byreason.most_common():
        if desc not in structural:
            print(f"      {n:>4} 题  {desc}")
    print("    结构化判据：")
    for desc in sorted(structural):
        print(f"      {byreason.get(desc, 0):>4} 题  {desc}")

    if n_struct:
        print("\n  选项重复的题：")
        for k, rs in hits.items():
            if any(r in structural for r in rs):
                n_opt = len(next(i for i in pool if i.id == k).options)
                n_uniq = len({o.strip() for o in next(i for i in pool if i.id == k).options})
                print(f"    {k:<16} {n_opt} 个选项 / {n_uniq} 个不同"
                      f"  （重复 {n_opt - n_uniq} 处）")

    if hits:
        subj = Counter(next(i for i in pool if i.id == k).stratum.get("subject") for k in hits)
        allsub = Counter(i.stratum.get("subject") for i in pool)
        print(f"\n  分学科影响（剔除后仍 ≥18 题的学科不标警）：")
        for s, n in subj.most_common():
            rest = allsub[s] - n
            flag = "  ⚠️ 剩余偏少" if rest < 18 else ""
            print(f"    {s:<18}{allsub[s]:>4} -> {rest:>4}{flag}")

    if args.dry_run:
        print("\n  [dry-run] 未写盘。")
        return 0

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ds,
        "pool_size": len(pool),
        "reason": "MMLU-ProX 中文版：含 LaTeX 公式的题提取可靠性无法保证（既定策略：不使用）；"
                  "另有 5 题选项在翻译后出现完全相同的重复项",
        "rule_fingerprint": exclusions.rule_fingerprint(),
        "patterns": [{"regex": p, "desc": d} for p, d in exclusions.BAD_PATTERNS],
        "structural_rules": [{"key": k, "desc": d} for k, d in exclusions.STRUCTURAL_RULES],
        "instance_ids": sorted(hits),
        "per_instance_reasons": {k: v for k, v in sorted(hits.items())},
        "effect": {
            "excluded": len(hits),
            "remaining": len(pool) - len(hits),
            "note": "题集文件不改动；分析层读本清单过滤，raw/tallies 完整保留，可随时撤销",
        },
    }
    p = exclusions.list_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已写 {p.relative_to(paths.ROOT)}")
    print(f"  规则指纹 {out['rule_fingerprint']}")
    print(f"  下一步：分析脚本读 `meval.exclusions`，题集文件**不动**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
