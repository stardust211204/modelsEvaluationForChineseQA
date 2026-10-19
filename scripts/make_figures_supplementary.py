"""补充实验图件（三张）—— 供正式报告 §3.5 / §3.6 / §3.9 使用。

产出
    42_thinking_switch.png      思考开关效应（§3.9）
    43_format_decomposition.png 自由作答增益拆解（§3.5）
    44_shot_gradient.png        5-shot 数量梯度（§3.6）

★ 样式沿用 make_figures_v3.py 的约定
    * **必须先 `plt.style.use(...)`，再覆盖 `font.sans-serif`**
      —— seaborn 会把 font.family 设成 Arial，而 Arial 没有中文字形，
         中文会渲染成豆腐块且**只报 UserWarning、不报错**。
    * figsize 用倍数 `(5 * 列数, 3.5 * 行数)`；`bbox_inches="tight"`
    * 模型名一律用**全写**（图上不出现 ds-flash 这类缩写）
    * 冷色调 5 级色阶

★ 数据口径（三张图各自的题集都经过统一）
    42：E1/E1b 的**同一批 76 题**（剔除任一个模型在 8192 下没做完的题）
    43：E6 四组的**共同可判集 255 条**（四组落在完全相同的题上）
    44：E5 各档的原始可判集（分母差异已在正文注明）

用法
    .venv\\Scripts\\python.exe scripts/make_figures_supplementary.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- 样式：先好 seaborn，再覆盖中文字体（顺序不能反）-----------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

from meval import exclusions, metrics, paths  # noqa: E402
from meval.types import ParseMode  # noqa: E402

OUT = paths.ROOT / "figures"
OUT.mkdir(exist_ok=True)

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
COOL5 = ["#243B4A", "#3D5A73", "#5C7A99", "#8AA3B8", "#B8C6D1"]
COOL_DARK, COOL_MID, COOL_LIGHT = "#2E4057", "#5C7A99", "#A8B8C8"
WARM = "#C1666B"
DROP = exclusions.load()


def tally(run, cfg, arm, mode=ParseMode.STRICT):
    g = {}
    p = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    if not p.exists():
        return g
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != mode.value or t["arm"] != arm or t["instance_id"] in DROP:
            continue
        g[t["instance_id"]] = t["tier"]
    return g


# ══════════════════════════════════════════════════════════════════════════
# 42 思考开关效应
# ══════════════════════════════════════════════════════════════════════════
def fig42() -> None:
    OFF, ON = "e1-think-none-5arm", "e1-think-high8192-5arm"
    SRC_ON = {a: ("e1b-longcat-think8192" if a == "longcat-2.5-preview-free" else ON)
              for a in ARMS}
    off = {a: tally(OFF, "c3", a) for a in ARMS}
    on = {a: tally(SRC_ON[a], "c3", a) for a in ARMS}

    ALL = set.intersection(*[set(v) for v in on.values()])
    BAD = {i for i in ALL for a in ARMS if on[a][i] in ("truncated", "invalid")}
    keep = sorted(ALL - BAD)

    def acc(t, a):
        c = sum(1 for i in keep if t[a].get(i) == "correct")
        w = sum(1 for i in keep if t[a].get(i) == "wrong")
        return c / (c + w) * 100 if c + w else float("nan")

    order = sorted(ARMS, key=lambda a: -acc(on, a))
    a_off = [acc(off, a) for a in order]
    a_on = [acc(on, a) for a in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    w = 0.38
    b1 = ax.bar(x - w / 2, a_off, w, label="关闭思考", color=COOL_LIGHT,
                edgecolor=COOL_MID, linewidth=0.6)
    b2 = ax.bar(x + w / 2, a_on, w, label="开启思考", color=COOL_DARK,
                edgecolor=COOL_DARK, linewidth=0.6)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width() / 2,
                        r.get_height()), ha="center", va="bottom", fontsize=9)
    for i, a in enumerate(order):
        d = a_on[i] - a_off[i]
        ax.annotate(f"+{d:.1f}", (i, max(a_off[i], a_on[i]) + 7),
                    ha="center", fontsize=11, fontweight="bold", color=WARM)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9.5)
    ax.set_ylabel("准确率 ↑")
    ax.set_ylim(0, 108)
    ax.set_title(f"思考开关的效应：开启后五模型一致大幅上升 ↑（同一批 {len(keep)} 题，"
                 f"关闭思考为基线）")
    # ★ 图例原先放在 upper left，正好压住第一组柱（ds-pro）上方的 Δ 标注（+36.8，
    #   也就是效应区间上界），图上只剩四个 Δ。改放**坐标区内的右上角** ——
    #   那一块没有柱子也没有标注（最高的 mimo 组 Δ 在 86 附近，图例在 96 以上），
    #   只是换位置，不动图的整体设计。
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "42_thinking_switch.png")
    plt.close(fig)
    print(f"  ✅ 42_thinking_switch.png   题集 {len(keep)}   效应 "
          + "  ".join(f"{a.split('-')[0]}={a_on[i]-a_off[i]:+.1f}"
                      for i, a in enumerate(order)))


# ══════════════════════════════════════════════════════════════════════════
# 43 自由作答增益拆解
# ══════════════════════════════════════════════════════════════════════════
def fig43() -> None:
    G = [("A", "e6-a-5arm", "只能一个字母"),
         ("B", "e6-b-5arm", "可写理由\n不要求末行"),
         ("C", "e6-c-5arm", "可写理由\n要求末行答案"),
         ("D", "e6-d-5arm", "再要求\n逐步推理")]
    T = {l: {a: tally(r, "c2", a, ParseMode.LENIENT) for a in ARMS}
         for l, r, _ in G}
    JUDG = ("correct", "wrong")
    common = set.intersection(*[
        {(a, i) for a in ARMS for i, v in T[l][a].items() if v in JUDG}
        for l, _, _ in G])
    common = sorted(common)
    # ★ 口径限制：E6 的 B 组里「五个模型都可判」的题是 **0 道** ——
    #   因为 B 不要求末行答案，总有个别模型解析不出。
    #   ⇒ 不存在「所有模型 × 所有组」的共同题集。
    #   这里用 **(模型,题) 交集**：每一个模型在四组之间题集一致（可比阶梯），
    #   但不同模型的题集不同 ⇒ 均值线只是各模型值的平均，不是共同题集上的测量。
    per_arm_n = {a: sum(1 for (a2, _) in common if a2 == a) for a in ARMS}
    ns = sorted(set(per_arm_n.values()))

    def acc(l, a):
        c = sum(1 for (a2, i) in common if a2 == a and T[l][a].get(i) == "correct")
        w = sum(1 for (a2, i) in common if a2 == a and T[l][a].get(i) == "wrong")
        return c / (c + w) * 100 if c + w else float("nan")

    # 五个模型在各组的准确率（共同题集）
    per = {l: [acc(l, a) for a in ARMS] for l, _, _ in G}
    mean = [float(np.mean(per[l])) for l, _, _ in G]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(G))
    for j, a in enumerate(ARMS):
        ax.plot(x, [per[l][j] for l, _, _ in G], marker="o", markersize=5,
                linewidth=1.4, color=COOL5[j], label=a, alpha=0.85)
    ax.plot(x, mean, marker="s", markersize=9, linewidth=3, color=WARM,
            label="五个模型均值", zorder=5)
    # ★ 本图**不标注数值**——
    #   本图的阶梯是「各模型在自己的可判集上的均值」（四组共同子集，n = 255），
    #   与报告表里的「逐对配对子集」（n = 276/267/594）口径不同，数值因此有差异：
    #   图上是 +16.5/+10.4/+0.5，表里是 +15.9/+10.0/−0.4。
    #   两个口径各自成立，但把两套数字并排放进同一份报告，读者会以为其中一处是错的。
    #   故图只讲结论（形状与趋势），数值一律以表为准。
    #   详见 docs/reports-fix.md §6.1。
    #
    #   历史实现（保留备查，勿启用）：
    #   for i in range(len(G) - 1):
    #       d = mean[i + 1] - mean[i]
    #       y = max(mean[i], mean[i + 1]) + 4
    #       ax.annotate(f"{d:+.1f}", ((x[i] + x[i + 1]) / 2, y), ...)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{d}" for l, _, d in G], fontsize=9.5)
    ax.set_ylabel("准确率 ↑（宽松口径）")
    # ★ 标题**不引用具体数字** —— 本图的阶梯是「各模型在自己的可判集上的均值」，
    #   与报告 §3.5 表里的「逐对配对子集」口径不同（+15.9/+10.0/−0.4 vs
    #   +16.5/+10.4/+0.5）。引用数字会造成两处不一致，故只讲结论。
    ax.set_title("自由作答增益的拆解：允许写理由与要求末行两步贡献相当，"
                 "显式要求逐步推理无额外收益\n"
                 f"（各模型在自己的可判集上，每个模型 {ns[0]}~{ns[-1]} 题；"
                 "B 组不存在五个模型共同可判的题，故均值线为各模型值的平均）",
                 fontsize=10.5)
    ax.legend(frameon=True, loc="lower right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "43_format_decomposition.png")
    plt.close(fig)
    print(f"  ✅ 43_format_decomposition.png  每个模型 {ns[0]}~{ns[-1]} 题  均值 "
          + " → ".join(f"{m:.1f}" for m in mean))


# ══════════════════════════════════════════════════════════════════════════
# 44 5-shot 数量梯度
# ══════════════════════════════════════════════════════════════════════════
def fig44() -> None:
    PLAN = {
        "MMLU-ProX": {0: ("e6-a-5arm", "c2"), 1: ("e5-mmlu-n1-5arm", "c5"),
                      3: ("e5-mmlu-n3-5arm", "c5"), 5: ("e5-mmlu-n5-5arm", "c5"),
                      10: ("e5-mmlu-n10-5arm", "c5")},
        "C-Eval": {0: ("e5-ceval-n0-5arm", "c2"), 1: ("e5-ceval-n1-5arm", "c5"),
                   3: ("e5-ceval-n3-5arm", "c5"), 5: ("e5-ceval-n5-5arm", "c5"),
                   10: ("e5-ceval-n10-5arm", "c5")},
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (ds, lv) in zip(axes, PLAN.items()):
        ns = sorted(lv)
        curves = {}
        for a in ARMS:
            ys = []
            for n in ns:
                run, cfg = lv[n]
                t = tally(run, cfg, a)
                c = sum(1 for v in t.values() if v == "correct")
                w = sum(1 for v in t.values() if v == "wrong")
                ys.append(c / (c + w) * 100 if c + w else float("nan"))
            curves[a] = ys
        for j, a in enumerate(ARMS):
            ax.plot(ns, curves[a], marker="o", markersize=4, linewidth=1.3,
                    color=COOL5[j], label=a, alpha=0.85)
        m = [float(np.mean([curves[a][i] for a in ARMS])) for i in range(len(ns))]
        ax.plot(ns, m, marker="s", markersize=7, linewidth=2.6, color=WARM,
                label="五个模型均值", zorder=5)
        ax.set_title(f"{ds}（0→10 shot：{m[-1]-m[0]:+.1f} 个百分点）", fontsize=11)
        ax.set_xlabel("示例数量 →")
        ax.set_xticks(ns)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("准确率 ↑")
    axes[1].legend(frameon=True, loc="lower right", fontsize=8)
    fig.suptitle("5-shot 的示例数量梯度：曲线平直，示例数量不构成变量", y=1.02)
    fig.savefig(OUT / "44_shot_gradient.png")
    plt.close(fig)
    print("  ✅ 44_shot_gradient.png")


if __name__ == "__main__":
    print("=" * 92)
    print("  生成补充实验图件")
    print("=" * 92)
    fig42()
    fig43()
    fig44()
    print("\n  产出目录：figures/")
