"""核实面试官「叙述不周全 #4」：抽样实验「先抽样、后剔除」是否破坏了学科配比。

报告 §2.2：42 道源缺陷题在 150 题与 100 题抽样中同样自动滤除，
实际有效计分题数为 136 与 93。
本脚本统计：被滤掉的 14 题（与 7 题）落在哪些学科，占比是否偏离原始抽样分布。
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meval import metrics  # noqa: E402
from meval.types import Config  # noqa: E402


def subjects(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            st = r.get("stratum") or {}
            out[r["id"]] = st.get("subject") or "?"
    return out


ev = subjects(ROOT / "data" / "eval_set.jsonl")
print(f"  eval_set 共 {len(ev)} 题")
print()

for label, run, cfg in (("150 题抽样（E6）", "e6-a-5arm", "c2"),
                        ("100 题抽样（C6）", "c6-5arm-v2", "c6")):
    kept = {r["instance_id"] for r in metrics.load_tallies(run, Config(cfg), drop_excluded=True)}
    allr = {r["instance_id"] for r in metrics.load_tallies(run, Config(cfg), drop_excluded=False)}
    drop = sorted(allr - kept)
    print(f"  --- {label} ---")
    print(f"    抽中 {len(allr)} 题，剔除 {len(drop)} 题，有效计分 {len(kept)} 题")
    d_cnt = Counter(ev.get(i, "?") for i in drop)
    k_cnt = Counter(ev.get(i, "?") for i in kept)
    a_cnt = Counter(ev.get(i, "?") for i in allr)
    print(f"    {'学科':<20}{'抽中':>6}{'剔除':>6}{'剔除占该学科':>14}")
    for s in sorted(a_cnt, key=lambda x: -a_cnt[x]):
        n_a, n_d = a_cnt[s], d_cnt.get(s, 0)
        if n_d:
            print(f"    {s:<20}{n_a:>6}{n_d:>6}{n_d/n_a*100:>13.1f}%")
    print(f"    剔除率（总体）: {len(drop)/len(allr)*100:.1f}%")
    print()
