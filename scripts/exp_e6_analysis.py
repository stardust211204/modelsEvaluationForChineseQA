"""E6 自由作答增益成因拆解分析。

四组递进，每组相对上一组**只改一处**，且四组落在**同一批 150 题**上：

    A  e6-a-5arm   c1_strict.txt      只能一个字母
    B  e6-b-5arm   e6_b_noreq.txt     可写理由，**不要求**末行答案
    C  e6-c-5arm   c2_free.txt        可写理由 + **要求**末行「答案：X」
    D  e6-d-5arm   e6_d_stepwise.txt  可写理由 + **要求逐步推理** + 要求末行

★ 口径：**必须用宽松口径**
    B/C/D 的回复里模型会写理由，**严格口径的 `bare` 阶梯匹配不到任何东西**
    ⇒ 严格口径下这三组几乎无「可判」条目（实测 accuracy = nan）。
    宽松口径的阶梯第一级就是 `bare`，**向下兼容严格**，四组因此可比。
    本脚本同时报 A 组的严格值作为交叉校验。

判据（逐步隔离）
    A→B  允许写字本身的效果
    B→C  末行格式要求的效果
    C→D  显式要求推理的效果
    A→D  合计（应与 §3.5 报的 +13.8~29.1pp 同量级）
"""

import json
import statistics
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, metrics, paths
from meval.types import ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
DROP = exclusions.load()

GROUPS = [("A", "e6-a-5arm"), ("B", "e6-b-5arm"),
          ("C", "e6-c-5arm"), ("D", "e6-d-5arm")]
DESC = {"A": "只能一个字母", "B": "可写·不要求末行",
        "C": "可写·要求末行", "D": "可写·要求推理·要求末行"}


def load(run, mode):
    g = {}
    p = paths.ROOT / "runs" / run / "tallies" / "c2" / "all.jsonl"
    if not p.exists():
        return g
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != mode.value or t["instance_id"] in DROP or t["arm"] not in ARMS:
            continue
        g[(t["arm"], t["instance_id"])] = t
    return g


T = {lbl: load(run, ParseMode.LENIENT) for lbl, run in GROUPS}   # ★ 主口径
T_STRICT = {lbl: load(run, ParseMode.STRICT) for lbl, run in GROUPS}
missing = [l for l, t in T.items() if not t]
if missing:
    print(f"  ★ 尚无 tally 的组：{missing} —— 先跑批")
    raise SystemExit(0)

print("=" * 104)
print("  E6 自由作答增益成因拆解 · 四组递进（宽松口径）")
print("=" * 104)
print("  已判分条数： " + "   ".join(f"{l}={len(t)}" for l, t in T.items()))


def acc(t, arm):
    c = sum(1 for (a, i), v in t.items() if a == arm and v["tier"] == "correct")
    w = sum(1 for (a, i), v in t.items() if a == arm and v["tier"] == "wrong")
    return (c / (c + w) * 100 if c + w else float("nan")), c + w


# ---- 表1：四组准确率 -------------------------------------------------------
print(f"\n  【表1】四组准确率（宽松口径）")
print(f"  {'模型':<10}" + "".join(f"{l:>9}" for l, _ in GROUPS) + f"{'A→D':>9}")
for arm in ARMS:
    row = [acc(T[l], arm)[0] for l, _ in GROUPS]
    print(f"  {SH[arm]:<10}" + "".join(f"{x:>8.1f}%" for x in row)
          + f"{row[3]-row[0]:>+8.1f}")

print(f"\n  【表1b】交叉校验：A 组（只答字母）的严格 vs 宽松")
print(f"  {'模型':<10}{'严格':>9}{'宽松':>9}{'差':>8}   说明")
for arm in ARMS:
    a, n1 = acc(T_STRICT["A"], arm)
    b, n2 = acc(T["A"], arm)
    print(f"  {SH[arm]:<10}{a:>8.1f}%{b:>8.1f}%{b-a:>+7.1f}   "
          f"{'一致 ✓' if abs(b-a) < 0.6 else '★ 有差异，需查'}")
print("  ⇒ A 组两者应几乎相同（回复本来就是裸字母）；若不同说明宽松口径多认了东西。")

print(f"\n  【表1c】B/C/D 在严格口径下的可判条目（说明为何必须换口径）")
for l, _ in GROUPS:
    ns = sum(1 for v in T_STRICT[l].values() if v["tier"] in ("correct", "wrong"))
    nl = sum(1 for v in T[l].values() if v["tier"] in ("correct", "wrong"))
    print(f"      {l}  严格可判 {ns:>4}   宽松可判 {nl:>4}"
          f"   {'★ 严格口径不可用' if ns < nl * 0.5 else ''}")


# ---- 表2：逐步配对 McNemar -------------------------------------------------
def paired(la, lb, arm):
    b = c = 0
    for iid in {i for (a, i) in T[la] if a == arm} & {i for (a, i) in T[lb] if a == arm}:
        x, y = T[la][(arm, iid)]["tier"], T[lb][(arm, iid)]["tier"]
        if x not in ("correct", "wrong") or y not in ("correct", "wrong"):
            continue
        if x == "correct" and y == "wrong":
            b += 1
        elif x == "wrong" and y == "correct":
            c += 1
    return b, c


print()
print("=" * 104)
print("  【表2】逐步隔离：相邻两组之间的配对 McNemar（同 150 题）")
print("=" * 104)
STEPS = [("A", "B", "A→B 允许写字"), ("B", "C", "B→C 要求末行答案"),
         ("C", "D", "C→D 要求逐步推理"), ("A", "D", "A→D 合计"),
         ("A", "C", "A→C（对照 §3.5）")]
for la, lb, title in STEPS:
    print(f"\n  ▸ {title}")
    print(f"    {'模型':<10}{'前':>8}{'后':>8}{'差':>8}{'b':>5}{'c':>5}{'p':>11}")
    tb = tc = 0
    ds_ = []
    for arm in ARMS:
        a1, _ = acc(T[la], arm)
        a2, _ = acc(T[lb], arm)
        b, c = paired(la, lb, arm)
        tb += b
        tc += c
        ds_.append(a2 - a1)
        p = metrics.mcnemar_exact(b, c) if b + c else float("nan")
        print(f"    {SH[arm]:<10}{a1:>7.1f}%{a2:>7.1f}%{a2-a1:>+7.1f}{b:>5}{c:>5}{p:>11.4f}")
    p = metrics.mcnemar_exact(tb, tc) if tb + tc else float("nan")
    print(f"    {'合并五个模型':<10}{'':>8}{'':>8}{statistics.mean(ds_):>+7.1f}"
          f"{tb:>5}{tc:>5}{p:>11.4g}")

# ---- 表3：解析阶梯与输出长度 ----------------------------------------------
print()
print("=" * 104)
print("  【表3】答案是怎么被解析出来的（宽松口径的 tier_step 分布）")
print("=" * 104)
for l, run in GROUPS:
    steps = Counter(v.get("tier_step") or "—" for v in T[l].values())
    print(f"  {l}（{DESC[l]}）  {dict(steps.most_common(6))}")

print()
print("  【表3b】输出长度（模型是否真的写了理由）")
R = {}
for l, run in GROUPS:
    d = defaultdict(list)
    for fp in (paths.ROOT / "runs" / run / "raw" / "c2").glob("*.jsonl"):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["arm"] in ARMS:
                d[r["arm"]].append(len(r.get("content") or ""))
    R[l] = d
print(f"  {'模型':<10}" + "".join(f"{l:>10}" for l, _ in GROUPS))
for arm in ARMS:
    print(f"  {SH[arm]:<10}"
          + "".join(f"{statistics.median(R[l][arm]) if R[l].get(arm) else 0:>10.0f}"
                    for l, _ in GROUPS))

print()
print("  读法：")
print("    · A→B 显著 ⇒ 增益主要来自「允许把推理写出来」本身")
print("    · B→C 显著 ⇒ 增益来自「末行必须给出答案」这一格式约束")
print("    · C→D 显著 ⇒ 增益来自「被显式要求逐步推理」")
print("    · 若 A→D 的量级与 §3.5 报的 +13.8~29.1pp 一致，则四组设计与既有结论自洽")
