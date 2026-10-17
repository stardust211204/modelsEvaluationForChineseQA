"""补齐面试官意见里"数据都有、只是没写进报告"的那几个数：

1. 各模型在主评测（c1 / C-Eval c1）上的 invalid 与 truncated 率 —— 报告 §2.4 承诺"单独统计"，
   但正文一处数值都没有。
2. 「合并」准确率的口径：确认是微观加权（两集答对总数 ÷ 两集总可判数），
   并给出两集实际权重（546 vs 583 并不相等）。
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meval import metrics  # noqa: E402
from meval.types import Config  # noqa: E402

BOARDS = (("MMLU-ProX", "c1-5arm-v2", "c1"), ("C-Eval", "ceval-c1-5arm-v2", "c1"))
ARMS = ["longcat-2.5-preview-free", "deepseek-v4.1-flash", "qwen3.8-flash",
        "deepseek-v4-pro", "mimo-v2.6-flash"]

# ---------- 1. invalid / truncated 率（严格口径，报告主表口径）----------
stat = {}          # arm -> board -> Counter
for label, run, cfg in BOARDS:
    for r in metrics.load_tallies(run, Config(cfg), drop_excluded=True):
        if r.get("mode") != "strict":
            continue
        d = stat.setdefault(r["arm"], {}).setdefault(label, defaultdict(int))
        d[r["tier"]] += 1

print("=" * 96)
print("  各模型 invalid / truncated 率（严格口径，已剔 42 道源缺陷题）")
print("=" * 96)
print(f"  {'模型':<28}{'题集':<11}{'可判':>6}{'invalid':>9}{'invalid率':>10}"
      f"{'truncated':>10}{'截断率':>8}")
for a in ARMS:
    for label, _, _ in BOARDS:
        d = stat[a][label]
        ok = d["correct"] + d["wrong"]
        iv, tr = d["invalid"], d["truncated"]
        n = ok + iv + tr
        print(f"  {a:<28}{label:<11}{ok:>6}{iv:>9}{iv/n*100:>9.2f}%{tr:>10}{tr/n*100:>7.2f}%")
    print()

# ---------- 2. 合并口径 ----------
print("=" * 96)
print("  「合并」准确率的口径核实（微观加权 vs 宏观平均）")
print("=" * 96)
print(f"  {'模型':<28}{'MMLU':>9}{'C-Eval':>9}{'微观加权':>10}{'宏观平均':>10}{'报告值':>9}")
REPORT = {"longcat-2.5-preview-free": 75.8, "deepseek-v4.1-flash": 74.7,
          "qwen3.8-flash": 74.0, "deepseek-v4-pro": 68.8, "mimo-v2.6-flash": 63.0}
for a in ARMS:
    m = stat[a]["MMLU-ProX"]
    c = stat[a]["C-Eval"]
    cm, wm = m["correct"], m["wrong"]
    cc, wc = c["correct"], c["wrong"]
    micro = (cm + cc) / (cm + wm + cc + wc) * 100
    macro = ((cm / (cm + wm)) + (cc / (cc + wc))) / 2 * 100
    print(f"  {a:<28}{cm/(cm+wm)*100:>8.1f}%{cc/(cc+wc)*100:>8.1f}%"
          f"{micro:>9.1f}%{macro:>9.1f}%{REPORT[a]:>8.1f}%")

n_m = sum(stat[ARMS[0]]["MMLU-ProX"][k] for k in ("correct", "wrong", "invalid", "truncated"))
n_c = sum(stat[ARMS[0]]["C-Eval"][k] for k in ("correct", "wrong", "invalid", "truncated"))
print()
print(f"  两集可判题数（以 {ARMS[0]} 计）：MMLU-ProX {n_m}、C-Eval {n_c}")
print(f"  ⇒ 微观加权下两集权重为 {n_m/(n_m+n_c)*100:.2f}% / {n_c/(n_m+n_c)*100:.2f}%，**并不相等**")
