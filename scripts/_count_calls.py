"""统计 runs/ 下的真实调用数，用于核对 §2.2 的「39,371 次有效调用」。

为什么要算
    如果补充实验要并进报告、成为「本项目评测设计的一部分」，
    那么 §2.2 的规模数字就必须把它们算进去，否则前后矛盾。

口径
    · 每次 (arm, instance, rep) 的 raw 记录算一次调用
    · 区分「有效」（拿到回复）与「失败」（连接错误等）
    · 同时按 run 前缀分组，便于看出哪些是本次并入的实验
"""

import json
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import paths

runs = paths.ROOT / "runs"
if not runs.exists():
    print("  无 runs/ 目录")
    raise SystemExit(1)

per_run = {}
for d in sorted(runs.iterdir()):
    if not d.is_dir():
        continue
    n = ok = 0
    for fp in d.rglob("*.jsonl"):
        if "raw" not in str(fp):
            continue
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            n += 1
            if r.get("success"):
                ok += 1
    if n:
        per_run[d.name] = (n, ok)

total = sum(v[0] for v in per_run.values())
total_ok = sum(v[1] for v in per_run.values())
print("=" * 96)
print("  runs/ 下的调用统计")
print("=" * 96)
print(f"  目录数 {len(per_run)}   记录数 {total:,}   成功 {total_ok:,}"
      f"   失败 {total - total_ok:,}")

# 按前缀分组
groups = defaultdict(lambda: [0, 0, 0])
for name, (n, ok) in per_run.items():
    key = re.sub(r"[-_].*$", "", name)
    g = groups[key]
    g[0] += 1
    g[1] += n
    g[2] += ok
print()
print(f"  {'前缀':<16}{'run 数':>7}{'记录':>10}{'成功':>10}")
for k, (c, n, ok) in sorted(groups.items(), key=lambda x: -x[1][1]):
    print(f"  {k:<16}{c:>7}{n:>10,}{ok:>10,}")

print()
print("  ★ 本次并入的实验（E1/E1b/E2/E2b/E3/E5/E6 与基线对照）：")
NEW = ("e1-", "e1b-", "e2-", "e2b-", "e3-", "e5-", "e6-", "c3-", "c5-")
new_n = sum(n for k, (n, ok) in per_run.items() if k.startswith(NEW))
new_ok = sum(ok for k, (n, ok) in per_run.items() if k.startswith(NEW))
print(f"     记录 {new_n:,}   成功 {new_ok:,}")
print(f"  其余（报告原有的批次）：记录 {total-new_n:,}   成功 {total_ok-new_ok:,}")
print()
print(f"  ⇒ 若全部计入：{total_ok:,} 次有效调用")
