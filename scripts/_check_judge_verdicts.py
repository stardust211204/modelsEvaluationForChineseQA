"""统计安全线 judge 的实际判定分布，核对 §2.7 的「三分类」表述。

prompts/judge_refusal.txt 要求 judge 输出 **四类**：
    REFUSE | PARTIAL | COMPLY | UNCLEAR
而报告 §2.7 写的是「judge 输出 REFUSE / PARTIAL / COMPLY **三分类**」。
本脚本查清 UNCLEAR 到底会不会出现、出现了怎么处理。
"""

import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import paths

PAT = re.compile(r"REFUSE|PARTIAL|COMPLY|UNCLEAR", re.IGNORECASE)

for rid in ("orbench-judge-full", "orbench-judge-qwen"):
    d = paths.ROOT / "runs" / rid / "raw" / "c7"
    if not d.exists():
        print(f"  {rid}: 无 raw/c7")
        continue
    first = Counter()
    allc = Counter()
    n = 0
    empty = 0
    for fp in sorted(d.glob("*.jsonl")):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            n += 1
            txt = r.get("response") or ""
            if not txt.strip():
                empty += 1
                continue
            hits = PAT.findall(txt)
            if hits:
                first[hits[0].upper()] += 1
            else:
                first["(无标记)"] += 1
            for h in set(x.upper() for x in hits):
                allc[h] += 1
    print("=" * 90)
    print(f"  {rid}   {n} 条   空回复 {empty}")
    print("=" * 90)
    print(f"  首个出现的标记（即解析器会取的那个）：")
    for k, v in first.most_common():
        print(f"      {k:<12}{v:>7,}   {v/max(n-empty,1)*100:>5.1f}%")
    print(f"  出现过的全部标记：{dict(allc)}")
    print()
