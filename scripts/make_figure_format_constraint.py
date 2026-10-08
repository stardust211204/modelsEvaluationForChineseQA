"""图 38 · 格式约束的代价（C2 自由作答 vs C1 限制格式作答）

风格与 make_figures_v3.py 一致：COOL5 冷色调、微软雅黑、dpi 130。
画法：配对哑铃图 —— 每个模型一条横线连「限制格式作答」与「自由作答」两点，
      右端标注提升幅度，让「一致上升」这个结论一眼可见。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from meval import exclusions, metrics, paths  # noqa: E402
from meval.types import Config  # noqa: E402

# ---- 全局样式（先样式后字体，否则中文字形丢失）------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

COOL5 = ["#243B4A", "#3D5A73", "#5C7A99", "#8AA3B8", "#B8C6D1"]
ARM_COLOR = {
    "longcat-2.5-preview-free": COOL5[0],
    "deepseek-v4.1-flash": COOL5[1],
    "qwen3.8-flash": COOL5[2],
    "deepseek-v4-pro": COOL5[3],
    "mimo-v2.6-flash": COOL5[4],
}
# ★ 图上不出现缩写，直接写全名（与 make_figures_v3.py 的约定一致）
ARMS = list(ARM_COLOR)
DROP = exclusions.load()

DATASETS = [
    ("MMLU-ProX", "eval_set.jsonl", ("c1-5arm-v2", "c1"), ("c2hi-5arm-v2", "c2")),
    ("C-Eval", "ceval_set.jsonl", ("ceval-c1-5arm-v2", "c1"), ("ceval-c2hi-5arm-v2", "c2")),
]


def tier_map(run, cfg, arm, mode="lenient"):
    ts = metrics.load_tallies(run, Config(cfg))
    return {t["instance_id"]: t["tier"] for t in ts
            if t["arm"] == arm and t["mode"] == mode
            and t["instance_id"] not in DROP
            and t["tier"] in ("correct", "wrong")}


def collect(dsfile, base, free):
    out = {}
    for a in ARMS:
        b = tier_map(*base, a)
        f = tier_map(*free, a)
        common = sorted(set(b) & set(f))
        accb = sum(1 for i in common if b[i] == "correct") / len(common) * 100
        accf = sum(1 for i in common if f[i] == "correct") / len(common) * 100
        out[a] = (accb, accf)
    return out


def main() -> int:
    # figsize 加宽：全名最长 24 字符（longcat-2.5-preview-free），缩写只需一半
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

    for ax, (dsname, dsfile, base, free) in zip(axes, DATASETS):
        data = collect(dsfile, base, free)
        order = sorted(ARMS, key=lambda a: data[a][0])   # 按限制格式作答升序
        ys = list(range(len(order)))

        for y, a in zip(ys, order):
            x0, x1 = data[a]
            c = ARM_COLOR[a]
            ax.plot([x0, x1], [y, y], color=c, linewidth=2.0,
                    solid_capstyle="round", alpha=0.55, zorder=1)
            ax.scatter([x0], [y], s=58, facecolor="white", edgecolor=c,
                       linewidth=1.8, zorder=3)
            ax.scatter([x1], [y], s=58, facecolor=c, edgecolor=c,
                       linewidth=1.8, zorder=3)
            ax.text(x1 + 1.1, y, f"+{x1-x0:.1f}", va="center", ha="left",
                    fontsize=10, color=c, fontweight="bold")

        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=9.2)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel("准确率（%）", fontsize=10.5)
        ax.set_title(dsname, fontsize=12, color="#243B4A", pad=9)
        ax.grid(axis="x", alpha=0.28, linewidth=0.7)
        ax.set_axisbelow(True)
        # 给右侧的标注留位置
        lo = min(min(v) for v in data.values())
        hi = max(max(v) for v in data.values())
        ax.set_xlim(lo - 3.5, hi + 7.5)

    # 图例：放在上图外
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor="white", markeredgecolor="#5C7A99",
               markeredgewidth=1.6, label="限制格式作答"),
        Line2D([], [], marker="o", linestyle="", markersize=8,
               markerfacecolor="#5C7A99", markeredgecolor="#5C7A99",
               label="自由作答"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.045))

    fig.suptitle("去掉格式约束后，五个模型的准确率一致上升",
                 fontsize=13, color="#243B4A", y=1.02)
    fig.tight_layout()

    dst = paths.ROOT / "figures" / "38_format_constraint.png"
    fig.savefig(dst)
    print(f"  已写出 {dst.relative_to(paths.ROOT)}")

    # 同时把数值落盘，便于报告核对
    payload = {}
    for dsname, dsfile, base, free in DATASETS:
        d = collect(dsfile, base, free)
        payload[dsname] = {a: {"受限格式": round(d[a][0], 1),
                                   "自由作答": round(d[a][1], 1),
                                   "提升": round(d[a][1] - d[a][0], 1)}
                           for a in ARMS}
    jp = paths.ROOT / "reports" / "c2_vs_c1.json"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写出 {jp.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
