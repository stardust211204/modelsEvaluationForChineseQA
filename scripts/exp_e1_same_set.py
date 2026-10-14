"""E1/E1b 的**同题集**对照：关思考 vs 开思考。

★ 为什么必须重算
    「开思考」的数字是在**统一 76 题**（剔除任一臂没做完的 17 题）上算的，
    而「关思考」此前一直是**逐臂 93 题**。
    **题目不同 ⇒ 两个数字不能相减。**

    本项目已有的证据正说明这会出错：被剔除的 17 题在其余四臂上只有
    57.7~73.1% 的正确率，而保留的题有 81~85% —— 差 9~28pp。
    拿 93 题的分母去比 76 题的分母，等于给"开思考"凭空加了偏差。

本脚本
    ① 用 8192 组定出统一题集（剔除任一臂截断/无响应的题）
    ② 把**关思考组也限制在同一批题**上
    ③ 同题集下重算五臂的「关思考 → 开思考」效应
"""

import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, metrics, paths
from meval.types import ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "qwen3.8-flash",
        "mimo-v2.6-flash", "longcat-2.5-preview-free"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "qwen3.8-flash": "qwen", "mimo-v2.6-flash": "mimo",
      "longcat-2.5-preview-free": "longcat"}
OFF = "e1-think-none-5arm"                       # 关思考 @1024
ON = {a: ("e1b-longcat-think8192" if a == "longcat-2.5-preview-free"
          else "e1-think-high8192-5arm") for a in ARMS}
DROP = exclusions.load()
BAD = ("truncated", "invalid")


def load(run, arm):
    tp = paths.ROOT / "runs" / run / "tallies" / "c3" / "all.jsonl"
    g = {}
    if not tp.exists():
        return g
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


OFFT = {a: load(OFF, a) for a in ARMS}
ONT = {a: load(ON[a], a) for a in ARMS}

# ---- ① 用 8192 组定统一题集 ------------------------------------------------
ALL = set.intersection(*[set(v) for v in ONT.values()])
DROPPED = {i for i in ALL for a in ARMS if ONT[a][i] in BAD}
KEEP = sorted(ALL - DROPPED)
print("=" * 100)
print("  ① 统一题集")
print("=" * 100)
print(f"  五臂共有计分题 {len(ALL)}   剔除（任一臂在 8192 下没做完）{len(DROPPED)}"
      f"   保留 {len(KEEP)}")

# ---- ② 同题集下的两条件准确率 ---------------------------------------------
print()
print("=" * 100)
print(f"  ② ★ 同一批 {len(KEEP)} 题上的「关思考 → 开思考」")
print("=" * 100)
print(f"  {'臂':<10}{'关思考':>9}{'开思考':>9}{'效应':>9}{'b':>5}{'c':>5}{'p':>11}"
      f"{'开思考 reasoning 中位':>22}")
res = {}
for a in ARMS:
    c0 = sum(1 for i in KEEP if OFFT[a].get(i) == "correct")
    w0 = sum(1 for i in KEEP if OFFT[a].get(i) == "wrong")
    c1 = sum(1 for i in KEEP if ONT[a][i] == "correct")
    w1 = sum(1 for i in KEEP if ONT[a][i] == "wrong")
    a0 = c0 / (c0 + w0) * 100 if c0 + w0 else float("nan")
    a1 = c1 / (c1 + w1) * 100 if c1 + w1 else float("nan")
    b = sum(1 for i in KEEP
            if OFFT[a].get(i) == "correct" and ONT[a][i] == "wrong")
    cc = sum(1 for i in KEEP
             if OFFT[a].get(i) == "wrong" and ONT[a][i] == "correct")
    p = metrics.mcnemar_exact(b, cc) if b + cc else float("nan")
    res[a] = (a0, a1, b, cc, p, c0 + w0, c1 + w1)
    print(f"  {SH[a]:<10}{a0:>8.1f}%{a1:>8.1f}%{a1-a0:>+8.1f}{b:>5}{cc:>5}{p:>11.4g}"
          f"{'':>22}")
print()
print("  注：'关思考'分母 = " + str({SH[a]: res[a][5] for a in ARMS}))
print("      '开思考'分母 = " + str({SH[a]: res[a][6] for a in ARMS}))
print("      ⇒ 两组同题，但关思考组在个别题上可能是 invalid/truncated ⇒ 分母略小")

# ---- ③ 与旧口径的对照 ------------------------------------------------------
print()
print("=" * 100)
print("  ③ 与「逐臂 93 题」旧口径的对照（说明为什么必须换）")
print("=" * 100)
print(f"  {'臂':<10}{'旧：关思考93题':>15}{'旧：开思考逐臂':>15}{'旧效应':>9}"
      f"{'新效应':>9}{'差':>8}")
OLD_ON = {"deepseek-v4.1-flash": 83.3, "deepseek-v4-pro": 87.3, "qwen3.8-flash": 82.8,
          "mimo-v2.6-flash": 73.9, "longcat-2.5-preview-free": 85.2}
OLD_OFF = {"deepseek-v4.1-flash": 57.0, "deepseek-v4-pro": 50.5, "qwen3.8-flash": 50.5,
           "mimo-v2.6-flash": 42.9, "longcat-2.5-preview-free": 46.7}
for a in ARMS:
    a0, a1 = res[a][0], res[a][1]
    print(f"  {SH[a]:<10}{OLD_OFF[a]:>14.1f}%{OLD_ON[a]:>14.1f}%"
          f"{OLD_ON[a]-OLD_OFF[a]:>+8.1f}{a1-a0:>+8.1f}"
          f"{(a1-a0)-(OLD_ON[a]-OLD_OFF[a]):>+7.1f}")

print()
print("  排序：")
o_off = sorted(ARMS, key=lambda a: -res[a][0])
o_on = sorted(ARMS, key=lambda a: -res[a][1])
print("    关思考：" + " > ".join(SH[a] for a in o_off))
print("    开思考：" + " > ".join(SH[a] for a in o_on))
