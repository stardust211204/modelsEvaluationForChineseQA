"""E1 思考开关分析：三组对照 + 逐臂推断。

组别
    1  e1-think-none-5arm        reasoning_effort 保持 arms.yaml 默认（none / thinking=disabled）
    2  e1-think-high1024-5arm    reasoning_effort=high，max_tokens=1024（同预算）
    3  e1-think-high8192-5arm    reasoning_effort=high，max_tokens=8192（足额预算）

判据
    1→2  同预算下「让它思考」值不值
    2→3  「预算」值多少（开思考时）
    1→3  合计效果
"""

import json
import statistics
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, metrics, paths
from meval.types import Config, ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
# longcat 的开关参数名是 thinking，三组恒为关思考 ⇒ 不进 1↔2 / 1↔3 的推断
REASON_ARMS = ["mimo-v2.6-flash", "deepseek-v4.1-flash", "qwen3.8-flash", "deepseek-v4-pro"]
GROUPS = [("none@1024", "e1-think-none-5arm"),
          ("high@1024", "e1-think-high1024-5arm"),
          ("high@8192", "e1-think-high8192-5arm")]
DROP = exclusions.load()


def raw_of(run):
    g = defaultdict(dict)
    for fp in (paths.ROOT / "runs" / run / "raw" / "c3").glob("*.jsonl"):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                g[(r["arm"], r["instance_id"])] = r
    return g


def tally_of(run):
    """(arm, iid) -> tier（严格口径，已剔排除项）"""
    g = {}
    p = paths.ROOT / "runs" / run / "tallies" / "c3" / "all.jsonl"
    if not p.exists():
        return g
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if (t["mode"] != ParseMode.STRICT.value or t["instance_id"] in DROP
                or t["arm"] not in ARMS):
            continue
        g[(t["arm"], t["instance_id"])] = t["tier"]
    return g


T = {label: tally_of(run) for label, run in GROUPS}
R = {label: raw_of(run) for label, run in GROUPS}

DONE = {label: len(ts) for label, ts in T.items()}
print("=" * 100)
print("  E1 思考开关 · 三组对照")
print("=" * 100)
print(f"  已判分条数： " + "   ".join(f"{k}={v}" for k, v in DONE.items()))
if any(v == 0 for v in DONE.values()):
    print("  ★ 有组尚未产出 tally，先等待跑批完成")
    raise SystemExit(0)

# ---- 表 1：准确率 + token 特征 -------------------------------------------
print(f"\n  【表1】逐臂：准确率（严格口径）与 token 特征")
print(f"  {'臂':<10}{'组':<12}{'准确率':>9}{'可判':>6}{'截断':>6}"
      f"{'reasoning均值':>14}{'out中位':>9}")
acc = {}
for arm in ARMS:
    for label, _ in GROUPS:
        ts = T[label]
        c = sum(1 for (a, i), v in ts.items() if a == arm and v == "correct")
        w = sum(1 for (a, i), v in ts.items() if a == arm and v == "wrong")
        tr = sum(1 for (a, i), v in ts.items() if a == arm and v == "truncated")
        acc[(arm, label)] = (c / (c + w) * 100 if c + w else float("nan"), c + w)
        rs = [r.get("reasoning_tokens") or 0 for (a, _), r in R[label].items() if a == arm]
        to = [r.get("tokens_out") or 0 for (a, _), r in R[label].items() if a == arm]
        if rs:
            print(f"  {SH[arm]:<10}{label:<12}{acc[(arm,label)][0]:>8.1f}%"
                  f"{c+w:>6}{tr:>6}{statistics.mean(rs):>14.1f}"
                  f"{statistics.median(to):>9.0f}")
    print()

# ---- 表 2：配对 McNemar ---------------------------------------------------
def paired(run_a, run_b, arms):
    out = {}
    for arm in arms:
        b = c = 0
        n = 0
        for iid in {i for (a, i) in T[run_a] if a == arm} & {i for (a, i) in T[run_b] if a == arm}:
            x, y = T[run_a][(arm, iid)], T[run_b][(arm, iid)]
            bx = x == "correct"
            by = y == "correct"
            if x not in ("correct", "wrong") or y not in ("correct", "wrong"):
                continue
            n += 1
            if bx and not by:
                b += 1
            elif by and not bx:
                c += 1
        out[arm] = (b, c, n, metrics.mcnemar_exact(b, c) if b + c else float("nan"))
    return out


print("=" * 100)
print("  【表2】配对 McNemar（同一批 100 题）")
print("=" * 100)
PAIRS = [("none@1024", "high@1024", "①→② 同预算下思考", REASON_ARMS),
         ("high@1024", "high@8192", "②→③ 预算（开思考时）", REASON_ARMS),
         ("none@1024", "high@8192", "①→③ 合计", REASON_ARMS),
         ("none@1024", "high@1024", "longcat（对照组·开关未变）",
          ["longcat-2.5-preview-free"])]
for a, b, title, arms in PAIRS:
    print(f"\n  ▸ {title}")
    print(f"    {'臂':<10}{'①→② 变化':>12}{'b(前对后错)':>13}{'c(前错后对)':>13}{'n':>6}{'p':>11}")
    for arm in arms:
        bb, cc, n, p = paired(a, b, [arm])[arm]
        d = acc[(arm, b)][0] - acc[(arm, a)][0]
        print(f"    {SH[arm]:<10}{d:>+11.1f}{bb:>13}{cc:>13}{n:>6}{p:>11.4f}")

# ---- 表 3：合并四臂 -------------------------------------------------------
print()
print("=" * 100)
print("  【表3】四臂合并（仅 REASON_ARMS）")
print("=" * 100)
for a, b, title, arms in PAIRS[:3]:
    bb = cc = 0
    for arm in arms:
        x, y, _, _ = paired(a, b, [arm])[arm]
        bb += x
        cc += y
    p = metrics.mcnemar_exact(bb, cc)
    da = statistics.mean(acc[(arm, b)][0] - acc[(arm, a)][0] for arm in arms)
    print(f"  {title:<26} 平均 {da:>+6.1f}pp   b={bb} c={cc}   p={p:.4g}"
          f"   {'★ 显著' if p < 0.05 else '不显著'}")

print()
print("  读法：")
print("    · ①→② 显著为正 ⇒ 同预算下思考有用")
print("    · ②→③ 显著为正 ⇒ 1024 预算把思考卡死了，原报告的低分部分来自预算")
print("    · 三者皆不显著   ⇒ 该端点上「开思考」形同虚设")
print("    · longcat 行是**阴性对照**：其开关三组未变，若它也有显著变化即为噪声基线")
