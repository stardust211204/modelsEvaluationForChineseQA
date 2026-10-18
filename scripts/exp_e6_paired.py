"""E6 的**配对子集**口径：把四组的准确率放到同一批题上。

问题
    E6 四组的可判条目数差别很大：A=680、B=276、C=635、D=612
    —— 因为 B 组不要求末行答案，宽松口径也常常解析不出。
    所以 A→B 的「62.5% → 70.9%」是**两个不同分母**上的比例，不能直接相减。

    这与 E1b 遇到的问题同类（那边是截断，这边是解析不出），
    解法也一样：**只在两组都可判的条目上比**。

本脚本
    对每一对相邻组，先取**共同可判子集**，在该子集上分别算两组的准确率，
    再算差 —— 这样差值是同一批题上的，可直接解读。
    同时保留 McNemar 的 b/c/p（它们本来就在配对子集上）。
"""

import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, metrics, paths
from meval.types import ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
GROUPS = [("A", "e6-a-5arm"), ("B", "e6-b-5arm"),
          ("C", "e6-c-5arm"), ("D", "e6-d-5arm")]
DROP = exclusions.load()


def load(run):
    g = {}
    p = paths.ROOT / "runs" / run / "tallies" / "c2" / "all.jsonl"
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != ParseMode.LENIENT.value or t["arm"] not in ARMS:
            continue
        if t["instance_id"] in DROP:
            continue
        g[(t["arm"], t["instance_id"])] = t["tier"]
    return g


T = {l: load(r) for l, r in GROUPS}
print("=" * 100)
print("  E6 四组的可判条目（宽松口径）")
print("=" * 100)
for l, _ in GROUPS:
    c = Counter(T[l].values())
    print(f"  {l}: 总 {len(T[l])}  可判 {c['correct']+c['wrong']}"
          f"  （correct {c['correct']} / wrong {c['wrong']} / "
          f"invalid {c['invalid']} / truncated {c['truncated']}）")

print()
print("=" * 100)
print("  ★ 配对子集口径：每一对只在「两组都可判」的题上比")
print("=" * 100)
STEPS = [("A", "B", "A→B 允许写理由"), ("B", "C", "B→C 要求末行答案"),
         ("C", "D", "C→D 要求逐步推理"), ("A", "C", "A→C（≈ 报告 §3.5 的两个条件）"),
         ("A", "D", "A→D 合计")]
for la, lb, title in STEPS:
    print(f"\n  ▸ {title}")
    print(f"    {'模型':<10}{'共同可判':>9}{'前':>9}{'后':>9}{'差':>9}{'b':>5}{'c':>5}{'p':>11}")
    tot_b = tot_c = 0
    ds = []
    for a in ARMS:
        common = [i for (a2, i) in T[la]
                  if a2 == a and T[la][(a, i)] in ("correct", "wrong")
                  and T[lb].get((a, i)) in ("correct", "wrong")]
        if not common:
            continue
        c0 = sum(1 for i in common if T[la][(a, i)] == "correct")
        c1 = sum(1 for i in common if T[lb][(a, i)] == "correct")
        a0 = c0 / len(common) * 100
        a1 = c1 / len(common) * 100
        b = sum(1 for i in common
                if T[la][(a, i)] == "correct" and T[lb][(a, i)] == "wrong")
        cc = sum(1 for i in common
                 if T[la][(a, i)] == "wrong" and T[lb][(a, i)] == "correct")
        tot_b += b
        tot_c += cc
        ds.append(a1 - a0)
        p = metrics.mcnemar_exact(b, cc) if b + cc else float("nan")
        print(f"    {SH[a]:<10}{len(common):>9}{a0:>8.1f}%{a1:>8.1f}%"
              f"{a1-a0:>+8.1f}{b:>5}{cc:>5}{p:>11.4f}")
    p = metrics.mcnemar_exact(tot_b, tot_c) if tot_b + tot_c else float("nan")
    print(f"    {'合并五个模型':<10}{'':>9}{'':>9}{'':>9}{sum(ds)/len(ds):>+8.1f}"
          f"{tot_b:>5}{tot_c:>5}{p:>11.4g}")

print()
print("  ⇒ 与旧口径（各组自己的分母）相比，这里的「差」是同一批题上的，可直接引用。")
