"""把 OR-Bench 转成本项目统一的题集格式。

来源：`bench-llms/or-bench`（HF）的 `or-bench-hard-1k.csv`（1,319 条）
与 `or-bench-toxic.csv`（655 条）。论文：ICML 2025，https://par.nsf.gov/biblio/10672460/media/xml

为什么用它而不是自建
    我们自建的 300 条中文拒答题**全部饱和**（三个模型两侧都 0~1.3%），测不出差异。
    OR-Bench-Hard 的构造方式是「**被 ≥3 个大模型拒绝**」—— 按**区分度**筛题，
    这正是我们缺的。自建题做不到这一点（我们的模型会拒答，生不出毒种来筛）。

⚠️ 但有一个真实风险
    Hard 集是 **2024 年**按当时的模型筛的。我们的三个模型是 2026 年的模型，
    **可能已经把过度拒答修掉了** ⇒ 对我们**也可能饱和**。
    所以先跑 100 条试水，确认它能区分三个模型再上全量。

两侧的意义（对应 OR-Bench 的权衡框架）
    hard  → should_refuse=False，**过度拒答**侧：安全但像有害的请求，拒了就是过度拒答
    toxic → should_refuse=True， **毒害拒答**侧：该拒的请求，答了就是危险服从
    ★ 论文的核心结论是二者**权衡**（Spearman 相关 0.89），
      所以必须画成 (x=过度拒答率, y=毒害拒答率) 的散点，理想位置在左上角。

用法
    .venv\\Scripts\\python.exe scripts\\build_orbench_set.py --pilot 100
    .venv\\Scripts\\python.exe scripts\\build_orbench_set.py            # 全量 1,974 条
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import paths  # noqa: E402

SEED = 20261004
SOURCES = (("orbench_hard_raw.csv", "hard", False, "orb-h"),
           ("orbench_toxic_raw.csv", "toxic", True, "orb-t"))


def load_csv(name: str) -> list[dict]:
    p = paths.data() / name
    if not p.exists():
        raise SystemExit(f"!! 未找到 {p}（先跑下载步骤）")
    return list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))


def build(pilot: int | None) -> list[dict]:
    rng = random.Random(SEED)
    items: list[dict] = []
    for fname, side, should_refuse, prefix in SOURCES:
        rows = load_csv(fname)
        # 按 category 分层，保证试水集不偏科
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cat[(r.get("category") or "unknown").strip()].append(r)
        for v in by_cat.values():
            rng.shuffle(v)

        if pilot:
            # 每类按比例配额，向上取整，最后截断到 pilot//2
            half = pilot // len(SOURCES)
            total = len(rows)
            want = {c: max(1, round(half * len(v) / total)) for c, v in by_cat.items()}
            picked: list[dict] = []
            for c, v in sorted(by_cat.items()):
                picked.extend(v[:want[c]])
            rng.shuffle(picked)
            picked = picked[:half]
        else:
            picked = [r for v in by_cat.values() for r in v]

        for i, r in enumerate(picked, 1):
            items.append({
                "id": f"{prefix}-{i:04d}",
                "side": side,
                "should_refuse": should_refuse,
                "cat": (r.get("category") or "unknown").strip(),
                "prompt": (r.get("prompt") or "").strip(),
                "src": "or-bench",
            })
    rng.shuffle(items)
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 OR-Bench 题集")
    ap.add_argument("--pilot", type=int, default=None,
                    help="只抽 N 条（两侧各 N/2），用于先验证区分度")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = build(args.pilot)
    out = args.out or ("orbench_pilot.jsonl" if args.pilot else "orbench_set.jsonl")
    payload = "\n".join(json.dumps(i, ensure_ascii=False, sort_keys=True) for i in items)
    p = paths.data() / out
    p.write_text(payload + "\n", encoding="utf-8")
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    print("=" * 78)
    print(f"OR-Bench 题集 -> data/{out}   hash={h}")
    print("=" * 78)
    print(f"  合计 {len(items)} 条")
    print(f"  两侧: {dict(Counter(i['side'] for i in items))}")
    for side in ("hard", "toxic"):
        cats = Counter(i["cat"] for i in items if i["side"] == side)
        print(f"  {side:<6} 分类: {dict(cats)}")
    print()
    print("  下一步：")
    print(f"    .venv\\Scripts\\python.exe scripts\\run_refusal.py \\")
    print(f"        --run-id orbench-pilot --items-file data/{out} --max-tokens 2048")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
