"""8192 统一口径：**共同的剔除集**。

用户定的规则
    1. 用 8192 的数据
    2. 统计**所有**模型上被截断的题
    3. 在**所有**模型上剔除，保持剔除的题目一致

为什么这比逐模型剔除更严谨
    逐模型剔除时各模型分母不同（79 ~ 93），分母小的那个恰好剔掉了最难的题 ⇒ 虚高。
    取**并集**后，凡是任何一个模型没做完的题，对所有模型都不计 ——
    于是**五个模型落在完全相同的题目上**，分母相同，可直接比较。

★ 剔除集 = 截断（`finish_reason=length`）∪ 无响应（请求失败）
  —— 两者都属"没拿到答案"，且都不是模型答错。

★ 残留问题（必须声明）
    共同剔除集解决了"题集不同"，但**没有解决"某些模型的缺失更多"**：
    被剔除的题对 qwen 是"答了但不计"，对 ds-pro 是"根本没答出来"。
    前者无害，后者意味着 ds-pro 在这批题上的能力**未被观测**。
    所以仍给出留一法插补作为**稳健性检查**（不替换主表）。
"""

import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, paths
from meval.types import ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "qwen3.8-flash",
        "mimo-v2.6-flash", "longcat-2.5-preview-free"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "qwen3.8-flash": "qwen", "mimo-v2.6-flash": "mimo",
      "longcat-2.5-preview-free": "longcat"}
SRC = {a: ("e1b-longcat-think8192" if a == "longcat-2.5-preview-free"
           else "e1-think-high8192-5arm") for a in ARMS}
DROP = exclusions.load()
BAD = ("truncated", "invalid")


def load(arm):
    tp = paths.ROOT / "runs" / SRC[arm] / "tallies" / "c3" / "all.jsonl"
    g = {}
    for ln in tp.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != ParseMode.STRICT.value or t["arm"] != arm:
            continue
        if t["instance_id"] in DROP:
            continue
        g[t["instance_id"]] = t["tier"]
    return g


T = {a: load(a) for a in ARMS}
ALL = set.intersection(*[set(v) for v in T.values()])

# ---- 共同剔除集 = 任一个模型上被截断或无响应的题 --------------------------------
DROPPED_UNION = {i for i in ALL for a in ARMS if T[a][i] in BAD}
KEEP = sorted(ALL - DROPPED_UNION)

print("=" * 104)
print("  ① 共同剔除集的构成")
print("=" * 104)
print(f"  五个模型共有计分题           {len(ALL)}")
print(f"  ★ 共同剔除（任一个模型没做完）{len(DROPPED_UNION)}")
print(f"  ★ 保留下来用于比较        {len(KEEP)}")
print()
print(f"  {'题':<16}" + "".join(f"{SH[a]:>10}" for a in ARMS) + f"{'剔除原因':>12}")
for i in sorted(DROPPED_UNION):
    who = [SH[a] for a in ARMS if T[a][i] in BAD]
    print(f"  {i:<16}" + "".join(f"{T[a][i][:8]:>10}" for a in ARMS)
          + f"{','.join(who):>12}")

# ---- 表：统一题集上的准确率 ------------------------------------------------
print()
print("=" * 104)
print("  ② ★ 统一题集（{} 题）上的准确率 —— 五个模型分母完全相同".format(len(KEEP)))
print("=" * 104)
print(f"  {'模型':<10}{'正确':>6}{'错误':>6}{'分母':>6}{'准确率':>9}   与逐模型口径的对照")
rows = {}
for a in ARMS:
    c = sum(1 for i in KEEP if T[a][i] == "correct")
    w = sum(1 for i in KEEP if T[a][i] == "wrong")
    acc = c / (c + w) * 100 if c + w else float("nan")
    rows[a] = acc
    # 逐模型口径（作对照）
    ca = Counter(T[a][i] for i in KEEP)          # 占位，下面单独算
    pc = sum(1 for i in ALL if T[a][i] == "correct")
    pw = sum(1 for i in ALL if T[a][i] == "wrong")
    old = pc / (pc + pw) * 100 if pc + pw else float("nan")
    print(f"  {SH[a]:<10}{c:>6}{w:>6}{c+w:>6}{acc:>8.1f}%   逐模型 {old:.1f}%（{old-acc:+.1f}pp）")

order = sorted(ARMS, key=lambda a: -rows[a])
print()
print("  排序：" + " > ".join(f"{SH[a]} {rows[a]:.1f}" for a in order))

# ---- 稳健性：留一法插补 ----------------------------------------------------
print()
print("=" * 104)
print("  ③ 稳健性检查：留一法插补（**估计值，不替换主表**）")
print("=" * 104)
print("  假设被剔除的题若做完，正确率 = 其余四个模型在同题上的正确率。\n")
adj = {}
for a in ARMS:
    c = sum(1 for i in KEEP if T[a][i] == "correct")
    w = sum(1 for i in KEEP if T[a][i] == "wrong")
    exp = 0.0
    for i in DROPPED_UNION:
        cs = [T[o][i] for o in ARMS if o != a and i in T[o]]
        cc = sum(1 for x in cs if x == "correct")
        ww = sum(1 for x in cs if x == "wrong")
        if cc + ww:
            exp += cc / (cc + ww)
    adj[a] = (c + exp) / (c + w + len(DROPPED_UNION)) * 100
    print(f"  {SH[a]:<10}统一题集 {rows[a]:>6.1f}%   插补后 {adj[a]:>6.1f}%"
          f"   {adj[a]-rows[a]:>+6.1f}pp")
o2 = sorted(ARMS, key=lambda a: -adj[a])
print()
print("  统一题集排序：" + " > ".join(SH[a] for a in order))
print("  插补后排序  ：" + " > ".join(SH[a] for a in o2))
top = o2[:4]
print(f"\n  插补后前四名跨度 {max(adj[a] for a in top)-min(adj[a] for a in top):.1f}pp"
      f" —— {'低于' if max(adj[a] for a in top)-min(adj[a] for a in top) < 5 else '高于'}判定阈值 5pp")
