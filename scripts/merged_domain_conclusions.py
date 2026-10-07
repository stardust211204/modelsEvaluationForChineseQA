"""合并视图下的分领域显著结论。"""

import json
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ceval_domain_map import domain_of as ce_domain
from meval import build_set, exclusions, metrics, paths

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
drop = exclusions.load()
TH = 5.0

dom = {}
for i in build_set.load_frozen(paths.data() / "ceval_set.jsonl"):
    dom[i.id] = ce_domain(i)
for i in build_set.load_frozen(paths.data() / "eval_set.jsonl"):
    dom[i.id] = i.stratum.get("subject", "other")

tier: dict[str, dict[str, str]] = defaultdict(dict)
for run, cfg in (("c1-5arm-v2", "c1"), ("ceval-c1-5arm-v2", "c1")):
    f = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t["mode"] == "strict" and t["instance_id"] not in drop and t["arm"] in ARMS:
            tier[t["arm"]][t["instance_id"]] = t["tier"]

bydom: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
for a in ARMS:
    for iid, tr in tier[a].items():
        bydom[dom[iid]][a][iid] = tr

print("=" * 96)
print("  合并视图 · 分领域显著结论（Δ>=5pp 且 p<0.05）")
print("=" * 96)
summary = []
for d in sorted(bydom, key=lambda k: -len(bydom[k][ARMS[0]])):
    cell = bydom[d]
    accs = {}
    for a in ARMS:
        v = [t for t in cell[a].values() if t in ("correct", "wrong")]
        if v:
            accs[a] = sum(1 for x in v if x == "correct") / len(v)
    if len(accs) < 5:
        continue
    order = sorted(accs, key=lambda a: -accs[a])
    n = len(cell[order[0]])
    sig = []
    for x, y in combinations(order, 2):
        dd = (accs[x] - accs[y]) * 100
        if dd < TH:
            continue
        common = set(cell[x]) & set(cell[y])
        bx = sum(1 for i in common if cell[x][i] == "correct" and cell[y][i] != "correct")
        by_ = sum(1 for i in common if cell[x][i] != "correct" and cell[y][i] == "correct")
        p = metrics.mcnemar_exact(bx, by_)
        if p < 0.05:
            sig.append((SH[x], SH[y], dd, p))
    line = " > ".join(f"{SH[a]} {accs[a]:.1%}" for a in order)
    print(f"\n  ▸ {d}（n={n}）")
    print(f"    {line}")
    if sig:
        for w, l, dd, p in sig[:4]:
            print(f"      ✅ {w} > {l}  Δ{dd:+.1f}pp  p={p:.1e}")
            summary.append((d, w, l, dd, p))
        if len(sig) > 4:
            print(f"      …另 {len(sig)-4} 组")
    else:
        print("      — 无显著配对")
print(f"\n  合计 {len(summary)} 条显著结论")
