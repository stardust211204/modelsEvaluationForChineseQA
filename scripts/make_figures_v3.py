"""竞品报告图件 v3 —— 采用 HELM 的绘图规范。

HELM 的绘图规范（从 create_plots.py 提取）
    * `sns.set_style("whitegrid")` —— 白网格底
    * 图标题里带方向箭头：`Accuracy ↑` / `Calibration error ↓`
    * figsize 一律用倍数：`(5 * 列数, 3.5 * 行数)`
    * `bbox_inches="tight"` 保存

★ 一个必须记住的顺序坑
    seaborn 样式会把 `font.family` 设成 Arial，而 Arial **没有中文字形** ——
    中文会全部渲染成豆腐块，而且**只报 UserWarning、不报错**。
    ⇒ 必须**先** `plt.style.use(...)`，**再**覆盖 `font.sans-serif`。

★ 模型名一律用全写
    图上不出现 `ds-flash` 这类缩写，直接写 `deepseek-v4.1-flash`。

用法
    .venv\\Scripts\\python.exe scripts\\make_figures_v3.py
    .venv\\Scripts\\python.exe scripts\\make_figures_v3.py --only 1,3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- 样式：先套 seaborn，再覆盖中文字体（顺序不能反）--------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

UP, DOWN = "\u2191", "\u2193"

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
# ---- 冷色调配色（基于 #2E4057 / #5C7A99 / #A8B8C8 扩展为 5 级梯度）----
# 用单色梯度而非彩虹色：竞品报告要克制，且梯度天然表达"排序"语义。
COOL5 = ["#243B4A", "#3D5A73", "#5C7A99", "#8AA3B8", "#B8C6D1"]
COOL_DARK = "#2E4057"
COOL_MID = "#5C7A99"
COOL_LIGHT = "#A8B8C8"
# ★ 颜色按**总体排名**分配（深 = 准确率高），而不是按模型字母序。
#   这样两个目的同时达成：① 排行图自然呈现深浅梯度 ② 同一模型在所有图里颜色一致
COLOR = {
    "longcat-2.5-preview-free": COOL5[0],
    "deepseek-v4.1-flash": COOL5[1],
    "qwen3.8-flash": COOL5[2],
    "deepseek-v4-pro": COOL5[3],
    "mimo-v2.6-flash": COOL5[4],
}
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 数据层
# --------------------------------------------------------------------------
def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def merged_tiers() -> dict[str, dict[str, str]]:
    """合并 view：{model: {instance_id: tier}}（严格口径，已排除坏题）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from meval import exclusions, paths  # noqa: PLC0415

    drop = exclusions.load()
    out: dict[str, dict[str, str]] = {a: {} for a in ARMS}
    for run, cfg in (("c1-5arm-v2", "c1"), ("ceval-c1-5arm-v2", "c1")):
        f = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("mode") == "strict" and t["instance_id"] not in drop and t["arm"] in out:
                out[t["arm"]][t["instance_id"]] = t["tier"]
    return out


def acc_of(tiers: dict[str, str]) -> float:
    v = [x for x in tiers.values() if x in ("correct", "wrong")]
    return sum(1 for x in v if x == "correct") / len(v) * 100 if v else float("nan")


# --------------------------------------------------------------------------
# ① 总体准确率 —— 横向条形（HELM leaderboard 风格）
# --------------------------------------------------------------------------
def fig01() -> None:
    from meval import paths  # noqa: PLC0415

    drop_mod = paths.ROOT  # 占位

    # 分别取两个评测集 + 合并
    res: dict[str, dict[str, float]] = {}
    for label, run, cfg, ds in (("合并（1171 实例）", None, None, None),
                                ("MMLU-ProX", "c1-5arm-v2", "c1", "eval_set"),
                                ("C-Eval", "ceval-c1-5arm-v2", "c1", "ceval_set")):
        m: dict[str, float] = {}
        for a in ARMS:
            if run is None:
                m[a] = float("nan")
            else:
                f = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
                from meval import exclusions  # noqa: PLC0415
                d = exclusions.load()
                rows = [json.loads(l)["tier"] for l in f.read_text(encoding="utf-8").splitlines()
                        if l.strip() and json.loads(l)["arm"] == a
                        and json.loads(l)["mode"] == "strict"
                        and json.loads(l)["instance_id"] not in d]
                m[a] = acc_of({str(i): t for i, t in enumerate(rows)})
        res[label] = m
    mt = merged_tiers()
    res["合并（1171 实例）"] = {a: acc_of(mt[a]) for a in ARMS}

    order = sorted(ARMS, key=lambda a: -res["合并（1171 实例）"][a])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5 * 1.6, 3.5 * 1.4))
    ax.barh(y, [res["合并（1171 实例）"][a] for a in order],
            color=[COLOR[a] for a in order], height=0.62)
    for i, a in enumerate(order):
        v = res["合并（1171 实例）"][a]
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"准确率 {UP}（%）", fontsize=11)
    ax.set_xlim(0, 88)
    ax.set_title("总体准确率 · 合并 1171 实例", fontsize=12)
    fig.savefig(OUT / "30_overall_barh.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ② 分领域热力图
# --------------------------------------------------------------------------
def fig02() -> None:
    sys.path.insert(0, str(_SCRIPTS))
    from ceval_domain_map import PROX14, domain_of  # noqa: PLC0415
    from meval import build_set, metrics, paths  # noqa: PLC0415

    dom = {}
    for i in build_set.load_frozen(paths.data() / "ceval_set.jsonl"):
        dom[i.id] = domain_of(i)
    for i in build_set.load_frozen(paths.data() / "eval_set.jsonl"):
        dom[i.id] = i.stratum.get("subject", "other")

    mt = merged_tiers()
    cells: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for a in ARMS:
        for iid, tier in mt[a].items():
            if iid in dom:
                cells[dom[iid]][a].append(tier)

    domains = [d for d in PROX14 if sum(len(cells[d][a]) for a in ARMS) >= 5]
    domains.sort(key=lambda d: -sum(len(cells[d][a]) for a in ARMS) // len(ARMS))
    # ★ 列序按**总体准确率**排（与排行图一致），而不是 ARMS 的字母序
    overall = merged_tiers()
    cols = sorted(ARMS, key=lambda a: -acc_of(overall[a]))
    M = np.array([[(acc_of(dict(enumerate(cells[d][a]))) if cells[d][a] else np.nan)
                   for a in cols] for d in domains])
    ns = [max(len(cells[d][a]) for a in ARMS) for d in domains]

    fig, ax = plt.subplots(figsize=(5 * 2.0, 3.5 * 1.9))
    im = ax.imshow(M, cmap="Blues", vmin=np.nanmin(M) - 6, vmax=np.nanmax(M) + 2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=28, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([f"{d}  ({n})" for d, n in zip(domains, ns)], fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=8.5)
    ax.set_title(f"分领域准确率 {UP}（%），括号内为实例数", fontsize=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(OUT / "31_domain_heatmap.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ③ 难度与模型差距（留一法）
# --------------------------------------------------------------------------
def fig03() -> None:
    mt = merged_tiers()
    ids = [i for i in mt[ARMS[0]] if all(i in mt[a] for a in ARMS)]
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    xs, spread = [], []
    for lo, hi in bins:
        vals, pts = [], []
        for a in ARMS:
            others = [b for b in ARMS if b != a]
            sel = []
            for i in ids:
                v = [1 if mt[b].get(i) == "correct" else 0
                     for b in others if mt[b].get(i) in ("correct", "wrong")]
                if len(v) < 3:
                    continue
                if lo <= sum(v) / len(v) < hi:
                    sel.append(i)
            if len(sel) < 5:
                continue
            vals.append(sum(1 for i in sel if mt[a][i] == "correct") / len(sel) * 100)
            pts.append(np.mean([np.mean([1 if mt[b].get(i) == "correct" else 0
                                         for b in others if mt[b].get(i) in ("correct", "wrong")])
                                for i in sel]))
        if len(vals) >= 3:
            xs.append(np.mean(pts))
            spread.append(max(vals) - min(vals))
    fig, ax = plt.subplots(figsize=(5 * 1.4, 3.5 * 1.2))
    ax.plot(xs, spread, "o-", color=COOL_DARK, lw=2, ms=8)
    ax.set_xlabel("实例难度（其余四模型平均正确率，%）", fontsize=11)
    ax.set_ylabel("模型间最大差距（百分点）", fontsize=11)
    ax.set_title("难度与模型间差距", fontsize=12)
    fig.savefig(OUT / "32_difficulty_spread.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ④ 成本效率（散点：token vs 准确率，标中位延迟）
# --------------------------------------------------------------------------
def fig04() -> None:
    from meval import exclusions, metrics, paths  # noqa: PLC0415

    drop = exclusions.load()
    fig, ax = plt.subplots(figsize=(5 * 1.6, 3.5 * 1.3))
    for a in ARMS:
        f = paths.ROOT / "runs" / "c2hi-5arm-v2" / "raw" / "c2" / f"{a}.jsonl"
        rs = [r for r in load_jsonl(f) if r.get("success")]
        if not rs:
            continue
        to = sum(r.get("tokens_out") or 0 for r in rs) / 1000
        lat = sorted(r.get("latency_s") or 0 for r in rs)
        tf = paths.ROOT / "runs" / "c2hi-5arm-v2" / "tallies" / "c2" / "all.jsonl"
        rows = [json.loads(l)["tier"] for l in tf.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["arm"] == a
                and json.loads(l)["mode"] == "lenient"
                and json.loads(l)["instance_id"] not in drop]
        acc = acc_of(dict(enumerate(rows)))
        ax.scatter(to, acc, s=150, color=COLOR[a], zorder=3)
        ax.annotate(f"{a}\n{lat[len(lat)//2]:.2f}s", (to, acc), textcoords="offset points",
                    xytext=(9, -6), fontsize=8.5)
    ax.set_xlabel("输出 token（千）", fontsize=11)
    ax.set_ylabel(f"准确率 {UP}（%）", fontsize=11)
    ax.set_title("成本效率 · 自由作答 / MMLU-ProX 150 实例", fontsize=12)
    ax.set_xlim(-6, 165)
    ax.set_ylim(72, 86)
    fig.savefig(OUT / "33_cost_scatter.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ⑤ 安全双向（双面板）
# --------------------------------------------------------------------------
def fig05() -> None:
    from meval import paths  # noqa: PLC0415

    data = {}
    for run in ("orbench-judge-full", "orbench-judge-qwen"):
        p = paths.ROOT / "runs" / run / "refusal_judge_report.json"
        if not p.exists():
            continue
        for arm, v in json.loads(p.read_text(encoding="utf-8"))["arms"].items():
            data[arm] = (v["over"]["refuse_over_rate"] * 100,
                         v["under"]["comply_unsafe_rate"] * 100)
    if not data:
        return
    order = sorted(data, key=lambda a: -data[a][0])
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(5 * 1.7 * 2, 3.5 * 1.25))
    for ax, idx, title, col in ((axes[0], 0, f"过度拒答率 {DOWN}（%）", COOL_DARK),
                                (axes[1], 1, f"危险服从率 {DOWN}（%）", COOL_MID)):
        b = ax.bar(x, [data[a][idx] for a in order], 0.6, color=col)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + max(data[a][idx] for a in order) * 0.02,
                    f"{r.get_height():.1f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=22, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylim(0, max(data[a][idx] for a in order) * 1.22)
    axes[0].set_ylabel("比率（%）", fontsize=11)
    fig.suptitle("安全双向取舍 · OR-Bench（1319 无害 + 655 有害）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "34_refusal_twosided.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ⑥ 两评测集配对差值
# --------------------------------------------------------------------------
def fig06() -> None:
    from meval import exclusions, paths  # noqa: PLC0415

    drop = exclusions.load()

    def one(run, a):
        f = paths.ROOT / "runs" / run / "tallies" / "c1" / "all.jsonl"
        rows = [json.loads(l)["tier"] for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["arm"] == a
                and json.loads(l)["mode"] == "strict"
                and json.loads(l)["instance_id"] not in drop]
        return acc_of(dict(enumerate(rows)))

    p = {a: one("c1-5arm-v2", a) for a in ARMS}
    c = {a: one("ceval-c1-5arm-v2", a) for a in ARMS}
    fig, ax = plt.subplots(figsize=(5 * 1.5, 3.5 * 1.55))
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    for x, y in combinations(ARMS, 2):
        dx, dy = p[x] - p[y], c[x] - c[y]
        same = (dx > 0) == (dy > 0)
        ax.scatter(dx, dy, s=110, color=COOL_DARK if same else COOL_LIGHT,
                   marker="o" if same else "X", zorder=3)
        ax.annotate(f"{x} − {y}", (dx, dy), textcoords="offset points",
                    xytext=(7, 5), fontsize=6.8)
    for v in (5, -5):
        ax.axvline(v, color="#c00", ls=":", lw=1, alpha=0.6)
        ax.axhline(v, color="#c00", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("MMLU-ProX 上的差值（百分点）", fontsize=11)
    ax.set_ylabel("C-Eval 上的差值（百分点）", fontsize=11)
    ax.set_title("两评测集配对差值（绿=同向，红叉=翻转）", fontsize=12)
    fig.savefig(OUT / "35_cross_dataset.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ⑦ C-Eval 分难度层级
# --------------------------------------------------------------------------
def fig07() -> None:
    from meval import build_set, exclusions, paths  # noqa: PLC0415

    drop = exclusions.load()
    ce = build_set.load_frozen(paths.data() / "ceval_set.jsonl")

    def lv(s: str) -> str:
        if s.startswith("middle_school"):
            return "初中"
        if s.startswith("high_school"):
            return "高中"
        if s.startswith("college"):
            return "大学"
        return "职业/专业"

    sub = {i.id: lv(i.stratum["subject"]) for i in ce}
    f = paths.ROOT / "runs" / "ceval-c1-5arm-v2" / "tallies" / "c1" / "all.jsonl"
    tier: dict[str, dict[str, str]] = defaultdict(dict)
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t["mode"] == "strict" and t["instance_id"] not in drop:
            tier[t["arm"]][t["instance_id"]] = t["tier"]
    levels = ["初中", "高中", "大学", "职业/专业"]
    M, ns = [], []
    for L in levels:
        ids = [i for i, l in sub.items() if l == L]
        ns.append(len(ids))
        M.append([acc_of({i: tier[a][i] for i in ids if i in tier[a]}) for a in ARMS])
    M = np.array(M)
    # ★ 按总体准确率重排列（与排行图、热力图一致）
    _ov = {a: np.nanmean(M[:, k]) for k, a in enumerate(ARMS)}
    cols7 = sorted(ARMS, key=lambda a: -_ov[a])
    M = np.array([[M[i, ARMS.index(a)] for a in cols7] for i in range(len(levels))])
    x = np.arange(len(levels))
    w = 0.16
    fig, ax = plt.subplots(figsize=(5 * 1.85, 3.5 * 1.2))
    for k, a in enumerate(cols7):
        ax.bar(x + (k - 2) * w, M[:, k], w, label=a, color=COLOR[a])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n({n})" for L, n in zip(levels, ns)], fontsize=10)
    ax.set_ylabel(f"准确率 {UP}（%）", fontsize=11)
    ax.set_ylim(0, 104)
    ax.set_title("C-Eval 分难度层级", fontsize=12)
    # ★ 图例放**轴外**下方：放轴内会压住最高那组柱子
    ax.legend(fontsize=8.5, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), columnspacing=1.2, handlelength=1.4)
    fig.savefig(OUT / "36_ceval_levels.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# ⑧ 配比敏感性
# --------------------------------------------------------------------------
def fig08() -> None:
    from meval import exclusions, paths  # noqa: PLC0415

    drop = exclusions.load()

    def one(run, a):
        f = paths.ROOT / "runs" / run / "tallies" / "c1" / "all.jsonl"
        rows = [json.loads(l)["tier"] for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["arm"] == a
                and json.loads(l)["mode"] == "strict"
                and json.loads(l)["instance_id"] not in drop]
        return acc_of(dict(enumerate(rows)))

    p = {a: one("c1-5arm-v2", a) for a in ARMS}
    c = {a: one("ceval-c1-5arm-v2", a) for a in ARMS}
    fig, ax = plt.subplots(figsize=(5 * 1.5, 3.5 * 1.2))
    xs = [i / 20 for i in range(21)]
    for a in ARMS:
        ax.plot([w * 100 for w in xs], [p[a] * (1 - w) + c[a] * w for w in xs],
                color=COLOR[a], lw=2, label=a)
    ax.set_xlabel("C-Eval 在合并集中的占比（%）", fontsize=11)
    ax.set_ylabel(f"准确率 {UP}（%）", fontsize=11)
    ax.set_title("总体排名随评测集配比变化", fontsize=12)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.savefig(OUT / "37_mix_sensitivity.png")
    plt.close(fig)


FIGURES = [
    ("1", fig01, "30_overall_barh.png"),
    ("2", fig02, "31_domain_heatmap.png"),
    ("3", fig03, "32_difficulty_spread.png"),
    ("4", fig04, "33_cost_scatter.png"),
    ("5", fig05, "34_refusal_twosided.png"),
    ("6", fig06, "35_cross_dataset.png"),
    ("7", fig07, "36_ceval_levels.png"),
    ("8", fig08, "37_mix_sensitivity.png"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="竞品报告图件 v3（HELM 风格）")
    ap.add_argument("--only", default=None, help="只画指定编号，逗号分隔")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else None
    print("=" * 78)
    print("  竞品图件 v3（HELM 风格）")
    print("=" * 78)
    for num, fn, name in FIGURES:
        if want and num not in want:
            continue
        try:
            fn()
            print(f"  OK   {name:<34}{(OUT / name).stat().st_size / 1024:>7.1f} KB")
        except Exception as e:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            print(f"  FAIL {name:<34}{type(e).__name__}: {str(e)[:56]}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
