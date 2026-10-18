"""E2b 对照分析：分离「字母↔内容对应被破坏」与「题面排版异常」。

三方对照（同一批 100 题、同一模板 `prompts/e2_labelshuffle.txt` 或 c1_strict）

    C1   `c1-5arm-v2`      标签顺序排列、内容原序        基准（题面自然）
    E2b  `e2b-relabel-5arm` 标签打乱、**金标槽位仍显示原字母**  只有「排版被打乱」
    E2   `e2-relabel-5arm`  标签打乱、金标字母也随机        「排版被打乱」+「对应被破坏」

分解
    C1 − E2b  = **排版异常效应**（题面看起来乱不乱）
    E2b − E2  = **字母↔内容对应被破坏的效应**（模型依赖表面线索）

★ 为什么 E2b 能隔离排版效应
    在 C1 与 E2b 里，金标内容都在**同一个槽位 g**、都显示**同一个字母 LETTERS[g]**。
    所以「偏好字母 A」与「偏好第 k 个槽位」的模型在两个条件下命中率**完全相同**。
    唯一变化的是**其余选项的标签乱不乱**。

★ 判分必须重判
    两个实验的 `options`/`gold_indices` 都保持原样，tally 的 `tier` 判的是
    「答的字母下标 == 金标下标」，而字母是乱序贴在槽位上的 ⇒ 必须用
        content_index = label_order.index(answer_index)
    重判。（E2 实测原判会差 +25.8 ~ +45.2pp。）
"""

import json
import statistics
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import build_set, exclusions, metrics, paths
from meval.types import ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
LETTERS = "ABCDEFGHIJ"
DROP = exclusions.load()

RUNS = {
    "C1": ("c1-5arm-v2", "c1", None),
    "E2b": ("e2b-relabel-5arm", "c1", "e2b_relabel_set.meta.json"),
    "E2": ("e2-relabel-5arm", "c1", "e2_relabel_set.meta.json"),
}
# 全部用 eval_set 的题面与金标做判分基准（两个实验的 options/gold 都未改）
POOL = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}


def load(run, cfg, metafile):
    """(arm, iid) -> (tier, answer_index)，已按 label_order 重映射并重判。"""
    ans = {}
    for fp in (paths.ROOT / "runs" / run / "raw" / cfg).glob("*.jsonl"):
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["arm"] in ARMS:
                ans[(r["arm"], r["instance_id"])] = r
    tp = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    if not tp.exists():
        return {}
    L = None
    if metafile:
        L = json.loads((paths.data() / metafile).read_text(encoding="utf-8"))["label_order"]
    out = {}
    for ln in tp.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != ParseMode.STRICT.value or t["instance_id"] in DROP:
            continue
        if t["arm"] not in ARMS or t["answer_index"] is None:
            continue
        iid = t["instance_id"]
        if iid not in POOL:
            continue
        inst = POOL[iid]
        ai = t["answer_index"]
        # ★ 重映射：字母 -> 内容下标（无 meta 时即原序，恒等）
        ci = L[iid].index(ai) if L else ai
        tier = "correct" if ci in inst.gold_indices else "wrong"
        out[(t["arm"], iid)] = {"tier": tier, "content_index": ci,
                                "letter_index": ai, "n_opt": len(inst.options)}
    return out


T = {k: load(*v) for k, v in RUNS.items()}
missing = [k for k, v in T.items() if not v]
if missing:
    print(f"  ★ 尚无 tally：{missing} —— 先跑批")
    raise SystemExit(0)

print("=" * 100)
print("  E2b 对照：排版异常 vs 字母↔内容对应被破坏")
print("=" * 100)
print("  已判分条数： " + "   ".join(f"{k}={len(v)}" for k, v in T.items()))


def acc(t, arm):
    c = sum(1 for (a, _), v in t.items() if a == arm and v["tier"] == "correct")
    w = sum(1 for (a, _), v in t.items() if a == arm and v["tier"] == "wrong")
    return (c / (c + w) * 100 if c + w else float("nan")), c + w


# ---- 表1：三方准确率 + 分解 -------------------------------------------------
print(f"\n  【表1】三方准确率与效应分解")
print(f"  {'模型':<10}{'C1 基准':>10}{'E2b 仅乱排版':>14}{'E2 乱排版+破对应':>18}"
      f"{'排版效应':>11}{'对应效应':>11}{'合计':>9}")
rows = {}
for arm in ARMS:
    a1, _ = acc(T["C1"], arm)
    ab, _ = acc(T["E2b"], arm)
    a2, _ = acc(T["E2"], arm)
    rows[arm] = (a1, ab, a2)
    print(f"  {SH[arm]:<10}{a1:>9.1f}%{ab:>13.1f}%{a2:>17.1f}%"
          f"{ab-a1:>+10.1f}{a2-ab:>+10.1f}{a2-a1:>+8.1f}")
m = [statistics.mean(rows[a][i] for a in ARMS) for i in range(3)]
print(f"  {'五个模型均值':<10}{m[0]:>9.1f}%{m[1]:>13.1f}%{m[2]:>17.1f}%"
      f"{m[1]-m[0]:>+10.1f}{m[2]-m[1]:>+10.1f}{m[2]-m[0]:>+8.1f}")

print(f"\n  读法：")
print(f"    · 若 E2b ≈ C1 ⇒ 排版异常**不构成**下降来源，E2 的下降全部来自「对应被破坏」")
print(f"    · 若 E2b < C1 ⇒ 两者都有，`排版效应` 那一列就是排版异常的贡献")

# ---- 表2：配对 McNemar -----------------------------------------------------
print()
print("=" * 100)
print("  【表2】配对 McNemar（同 100 题）")
print("=" * 100)
for la, lb, title in (("C1", "E2b", "C1 → E2b（只有排版被打乱）"),
                      ("E2b", "E2", "E2b → E2（对应被破坏）"),
                      ("C1", "E2", "C1 → E2（合计）")):
    print(f"\n  ▸ {title}")
    print(f"    {'模型':<10}{'前':>8}{'后':>8}{'差':>8}{'b':>5}{'c':>5}{'p':>11}")
    tb = tc = 0
    ds = []
    for arm in ARMS:
        a1, _ = acc(T[la], arm)
        a2, _ = acc(T[lb], arm)
        b = c = 0
        for iid in {i for (a, i) in T[la] if a == arm} & {i for (a, i) in T[lb] if a == arm}:
            x, y = T[la][(arm, iid)]["tier"], T[lb][(arm, iid)]["tier"]
            if x == "correct" and y == "wrong":
                b += 1
            elif x == "wrong" and y == "correct":
                c += 1
        tb += b
        tc += c
        ds.append(a2 - a1)
        p = metrics.mcnemar_exact(b, c) if b + c else float("nan")
        print(f"    {SH[arm]:<10}{a1:>7.1f}%{a2:>7.1f}%{a2-a1:>+7.1f}{b:>5}{c:>5}{p:>11.4f}")
    p = metrics.mcnemar_exact(tb, tc) if tb + tc else float("nan")
    print(f"    {'合并五个模型':<10}{'':>8}{'':>8}{statistics.mean(ds):>+7.1f}"
          f"{tb:>5}{tc:>5}{p:>11.4g}")

# ---- 表3：字母偏好是否仍存在 -----------------------------------------------
print()
print("=" * 100)
print("  【表3】答「A」的比例（字母偏好的直接指标）")
print("=" * 100)
print(f"  {'模型':<10}{'C1':>9}{'E2b':>9}{'E2':>9}   说明")
for arm in ARMS:
    r = {}
    for k in ("C1", "E2b", "E2"):
        n = sum(1 for (a, _) in T[k] if a == arm)
        na = sum(1 for (a, _), v in T[k].items() if a == arm and v["letter_index"] == 0)
        r[k] = na / n * 100 if n else float("nan")
    print(f"  {SH[arm]:<10}{r['C1']:>8.1f}%{r['E2b']:>8.1f}%{r['E2']:>8.1f}%"
          f"   {'字母偏好稳定' if abs(r['E2b']-r['E2'])<5 else '★ 两条件不同'}")

print()
print("  说明：C1 与 E2b 的「答 A」比例应接近（同一套字母偏好）；")
print("        E2 也应接近 —— 打乱不改变模型爱答哪个字母，只改变那个字母指向谁。")
