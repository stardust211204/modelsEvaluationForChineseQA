"""E1b 分析：longcat 的思考效应（补上 E1 的缺口）。

背景
    E1 里 longcat 的思考**全程未打开** —— 三组都发了
    `thinking: {"type": "disabled"}`，`reasoning_tokens` 与 `reasoning` 正文均为 0。
    它在那张表里是**阴性对照**，不是"longcat 开思考提升小"。

    longcat 的开关与其余四臂不同：参数名是 `thinking`、值是嵌套字典。
    2026-10-03 的探测只试过**三种关思考写法**，从未试过开思考。
    重看原始探测数据才发现：**不传 `thinking` 时 longcat 默认就在思考**
    （`reasoning` 正文 416 字符、输出 96 token；传了 disabled 后是 0 字符、2 token）。

    ⇒ 打开方式就是 `--drop-param thinking`（把它从 params 里删掉，走默认）。

本脚本对比
    ① `e1-think-none-5arm`       longcat 关思考 @1024（E1 组①）
    ② `e1-think-high8192-5arm`   longcat 关思考 @1024（E1 组③，同参数、不同时段）
    ③ `e1b-longcat-think1024`    longcat 开思考 @1024  ← 与 ① 同预算，**纯单因子**
    ④ `e1b-longcat-think8192`    longcat 开思考 @8192  ← 与 E1 组③ 的其余四臂同预算

★ 两个方向相反的偏差，必须同时声明
    ③ 有 **32% 截断** ⇒ 截断项不进分母，而它们系统性偏难 ⇒ 准确率被**高估**
    ④ 有 **6 条连接失败**（远端 ~119s 掐断）⇒ 缺的正是推理最长的题 ⇒ 也**高估**
    ⇒ 两者都是**上界**；真实效应可能更大（因为难题被剔掉了）。
"""

import json
import statistics
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import exclusions, metrics, paths
from meval.types import ParseMode

ARM = "longcat-2.5-preview-free"
DROP = exclusions.load()
C3 = "c3"

GROUPS = [
    ("① 关思考 @1024", "e1-think-none-5arm", 1024),
    ("② 关思考 @1024（组③·异时段）", "e1-think-high8192-5arm", 1024),
    ("③ 开思考 @1024", "e1b-longcat-think1024", 1024),
    ("④ 开思考 @8192", "e1b-longcat-think8192", 8192),
]


def load(run):
    """(iid) -> (tier, reasoning_len, tokens_out, finish)"""
    raw = {}
    for fp in (paths.ROOT / "runs" / run / "raw" / C3).glob("*.jsonl"):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["arm"] == ARM:
                raw[r["instance_id"]] = r
    tp = paths.ROOT / "runs" / run / "tallies" / C3 / "all.jsonl"
    tier = {}
    if tp.exists():
        for ln in tp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            t = json.loads(ln)
            if t["mode"] == ParseMode.STRICT.value and t["arm"] == ARM:
                tier[t["instance_id"]] = t["tier"]
    out = {}
    for iid, r in raw.items():
        out[iid] = {
            "tier": tier.get(iid),
            "ok": bool(r.get("success")),
            "reasoning_len": len(r.get("reasoning") or ""),
            "tokens_out": r.get("tokens_out") or 0,
            "finish": r.get("finish_reason"),
        }
    return out


G = {lbl: load(run) for lbl, run, _ in GROUPS}
print("=" * 104)
print("  E1b · longcat 思考效应（补 E1 缺口）")
print("=" * 104)

# ---- 表1：总体 -------------------------------------------------------------
print(f"\n  【表1】longcat 在各条件下的表现（严格口径，已剔排除项）")
print(f"  {'条件':<28}{'准确率':>9}{'可判':>6}{'截断':>6}{'连接失败':>9}"
      f"{'reasoning中位':>14}{'输出tok中位':>12}")
acc = {}
for lbl, run, mt in GROUPS:
    g = G[lbl]
    c = sum(1 for i, v in g.items() if v["tier"] == "correct" and i not in DROP)
    w = sum(1 for i, v in g.items() if v["tier"] == "wrong" and i not in DROP)
    tr = sum(1 for i, v in g.items() if v["tier"] == "truncated" and i not in DROP)
    fail = sum(1 for v in g.values() if not v["ok"])
    rl = [v["reasoning_len"] for v in g.values()]
    to = [v["tokens_out"] for v in g.values()]
    a = c / (c + w) * 100 if c + w else float("nan")
    acc[lbl] = (a, c + w)
    print(f"  {lbl:<28}{a:>8.1f}%{c+w:>6}{tr:>6}{fail:>9}"
          f"{statistics.median(rl):>14.0f}{statistics.median(to):>12.0f}")

print(f"\n  读法：")
print(f"    · ① vs ③ —— **同预算（1024）、同题集、唯一差别是思考开关**")
print(f"    · ④ 与 E1 组③ 的其余四臂同预算（8192），可直接并列比较")

# ---- 表2：配对 McNemar -----------------------------------------------------
print()
print("=" * 104)
print("  【表2】配对 McNemar（同 100 题，只在两组都可判的条目上做）")
print("=" * 104)
PAIRS = [("① 关思考 @1024", "③ 开思考 @1024", "★ 单因子：思考开关（同 1024 预算）"),
         ("② 关思考 @1024（组③·异时段）", "④ 开思考 @8192", "对照 E1 组③ 的其余四臂"),
         ("① 关思考 @1024", "④ 开思考 @8192", "合计")]
for la, lb, title in PAIRS:
    b = c = 0
    for iid in set(G[la]) & set(G[lb]):
        x, y = G[la][iid]["tier"], G[lb][iid]["tier"]
        if x not in ("correct", "wrong") or y not in ("correct", "wrong"):
            continue
        if x == "correct" and y == "wrong":
            b += 1
        elif x == "wrong" and y == "correct":
            c += 1
    p = metrics.mcnemar_exact(b, c) if b + c else float("nan")
    a1, n1 = acc[la]
    a2, n2 = acc[lb]
    print(f"\n  ▸ {title}")
    print(f"      {a1:.1f}%（{n1} 可判） → {a2:.1f}%（{n2} 可判）   {a2-a1:+.1f}pp")
    print(f"      b（前对后错）={b}   c（前错后对）={c}   p={p:.4g}"
          f"   {'★ 显著' if p < 0.05 else '不显著'}")

# ---- 表3：与 E1 其余四臂并列 ----------------------------------------------
print()
print("=" * 104)
print("  【表3】★ 补上 longcat 后，E1 的「开思考」表就完整了")
print("=" * 104)
ARMS = ["deepseek-v4-pro", "deepseek-v4.1-flash", "qwen3.8-flash",
        "mimo-v2.6-flash", "longcat-2.5-preview-free"]
SH = {"deepseek-v4-pro": "ds-pro", "deepseek-v4.1-flash": "ds-flash",
      "qwen3.8-flash": "qwen", "mimo-v2.6-flash": "mimo",
      "longcat-2.5-preview-free": "longcat"}


def arm_acc(run, arm, mt):
    tp = paths.ROOT / "runs" / run / "tallies" / C3 / "all.jsonl"
    c = w = 0
    for ln in tp.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != ParseMode.STRICT.value or t["arm"] != arm:
            continue
        if t["instance_id"] in DROP:
            continue
        if t["tier"] == "correct":
            c += 1
        elif t["tier"] == "wrong":
            w += 1
    return c / (c + w) * 100 if c + w else float("nan"), c + w


print(f"  {'臂':<10}{'① 关思考':>10}{'③ 开思考@8192':>15}{'差':>9}   来源")
rows = []
for arm in ARMS:
    a0, _ = arm_acc("e1-think-none-5arm", arm, 1024)
    if arm == ARM:
        a1, _ = arm_acc("e1b-longcat-think8192", arm, 8192)
        src = "★ 本次补跑"
    else:
        a1, _ = arm_acc("e1-think-high8192-5arm", arm, 8192)
        src = "E1 组③"
    rows.append((arm, a0, a1))
    print(f"  {SH[arm]:<10}{a0:>9.1f}%{a1:>14.1f}%{a1-a0:>+8.1f}   {src}")

print()
print("  排序对照：")
print("    关思考：" + " > ".join(
    f"{SH[a]}" for a, _, _ in sorted(rows, key=lambda x: -x[1])))
print("    开思考：" + " > ".join(
    f"{SH[a]}" for a, _, _ in sorted(rows, key=lambda x: -x[2])))
