# -*- coding: utf-8 -*-
"""对照 report_data_provenance.md，检查 report_external_helm_polished.md 的
表格结构、图片引用、字符安全、交叉引用与关键数字是否自洽。"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "report_external_helm_polished.md"
OLD = ROOT / "reports" / "report_external_helm.md"

text = REP.read_text(encoding="utf-8")
old = OLD.read_text(encoding="utf-8")
lines = text.split("\n")

problems = []
notes = []


def bad(tag, msg):
    problems.append(f"[{tag}] {msg}")


def note(tag, msg):
    notes.append(f"[{tag}] {msg}")


# ---------- 1. 图片引用 ----------
img_old = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", old)
img_new = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
widths_new = re.findall(r"^\{width=(\d+%)\}$", text, re.M)
if not img_new:
    bad("图", f"润色版丢失了全部 {len(img_old)} 处图片引用，只剩 {len(widths_new)} 个 {{{{width=..%}}}} 占位")
    missing = sorted(set(img_old))
    note("图", "原报告引用：" + "、".join(missing))
else:
    lost = sorted(set(img_old) - set(img_new))
    if lost:
        bad("图", "丢失引用：" + "、".join(lost))

# 图号与 width 顺序比对：润色版按 §5.1 图表索引的顺序给出占位
new_order = widths_new
old_w = dict(re.findall(r"\(figures/([^)]+)\)\{width=(\d+%)\}", old))
idx = re.findall(r"\| `(\d+_[a-z_]+\.png)` \| §", text)
if len(idx) == len(new_order):
    mismatch = [(f, old_w.get(f), w) for f, w in zip(idx, new_order) if old_w.get(f) != w]
    if mismatch:
        for f, w1, w2 in mismatch:
            bad("图", f"{f} 宽度原为 {w1}，润色版占位为 {w2}")
    else:
        note("图", f"{len(new_order)} 处 width 占位与 §5.1 图表索引的图序、原报告宽度逐一吻合")
else:
    bad("图", f"图表索引 {len(idx)} 行 vs 正文 {len(new_order)} 处占位，数量不符")

# ---------- 2. 表格结构 ----------
def split_row(r):
    r = r.strip()
    if r.startswith("|"):
        r = r[1:]
    if r.endswith("|"):
        r = r[:-1]
    # 未被转义的竖线
    return re.split(r"(?<!\\)\|", r)


tables = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith("|") and i + 1 < len(lines) and re.match(
        r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
    ):
        start = i
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append((i + 1, lines[i]))
            i += 1
        tables.append((start + 1, rows))
    else:
        i += 1

for start, rows in tables:
    hdr_n = len(split_row(rows[0][1]))
    sep_n = len(split_row(rows[1][1]))
    if hdr_n != sep_n:
        bad("表", f"第 {start} 行表格：表头 {hdr_n} 列 / 分隔行 {sep_n} 列")
    for ln, r in rows[2:]:
        n = len(split_row(r))
        if n != hdr_n:
            bad("表", f"第 {ln} 行 {n} 列，表头为 {hdr_n} 列")
    if "<br>" in rows[0][1]:
        bad("表", f"第 {start} 行表头含 <br>：LaTeX 输出会丢行或原样输出 `<br>`")

note("表", f"共 {len(tables)} 张表")

# ---------- 3. 字符安全（微软雅黑缺字） ----------
UNSAFE = "⇒⊂⊃⊕↔⇔⟶"
for ch in UNSAFE:
    if ch in text:
        bad("字", f"含不可渲染字符 U+{ord(ch):04X} {ch}")

# ---------- 4. 交叉引用 ----------
heads = re.findall(r"^#{1,3} (\d+(?:\.\d+)?) ", text, re.M)
head_set = set(heads)
for m in re.finditer(r"§(\d+(?:\.\d+)?)", text):
    ref = m.group(1)
    if ref not in head_set:
        ln = text[: m.start()].count("\n") + 1
        bad("引", f"第 {ln} 行引用 §{ref}，但报告无该节（现有：{'、'.join(heads)}）")

# ---------- 5. 关键数字必须出现 ----------
MUST = {
    "53,379": "总调用数",
    "53,291": None,  # 润色版改成「逾 5.2 万」，见下
    "588": "MMLU-ProX 构建题数",
    "546": "MMLU-ProX 计分题数",
    "583": "C-Eval 题数",
    "1,974": "OR-Bench 题数",
    "1,319": "Hard 子集",
    "655": "Toxic 子集",
    "7,870": "judge 判定总数",
    "75.8%": "longcat 合并准确率",
    "62.9%": "longcat MMLU-ProX",
    "88.0%": "longcat C-Eval",
    "63.0%": "mimo 合并",
    "45.2%": "longcat 翻转率",
    "18.0%": "longcat C-Eval 翻转率",
    "28.0%": "mimo 稳定性",
    "87.0%": "longcat 稳定性",
    "6.19": "longcat 比值 MMLU",
    "8.19": "longcat 比值 C-Eval",
    "11.47": "ds-pro 比值 C-Eval",
    "1.31": "mimo 比值 MMLU",
    "1.21": "mimo 比值 C-Eval",
    "−8.5": "排版效应均值",
    "−3.1": "对应效应均值",
    "−11.6": "合计效应",
    "0.013": "合计 p 值",
    "−16.1": "ds-pro 对应效应",
    "0.004": "ds-pro 对应 p",
    "+29.1": "mimo 格式增益",
    "+13.8": "qwen 格式增益",
    "+15.9": "A→B",
    "+10.0": "B→C",
    "−0.4": "C→D",
    "+21.3": "A→C",
    "+2.8": "5-shot MMLU",
    "+0.9": "5-shot C-Eval",
    "82.4%": "ds-pro 自由作答",
    "83.5%": "ds-flash 自由作答",
    "30,113": "ds-pro token",
    "144,322": "qwen token",
    "2.87s": "ds-pro 延迟",
    "15.68s": "mimo 延迟",
    "82.5%": "qwen 过度拒答",
    "46.6%": "mimo 过度拒答",
    "8.6%": "mimo 危险服从",
    "1.4%": "qwen 危险服从",
    "51.3%": "ds-pro 关闭思考",
    "88.2%": "ds-pro 开启思考",
    "61.8%": "ds-flash 关闭思考",
    "+36.8": "ds-pro 思考效应",
    "+25.0": "ds-flash 思考效应",
    "28.1": "截断偏差",
    "10.5": "ds-pro 落后百分点（76 题口径）",
    "6.5": "ds-pro 落后百分点（93 题口径）",
    "76": "共同题集",
}
for k, label in MUST.items():
    if k not in text:
        if k == "53,291":
            note("数", "润色版未直接写 53,291，改写为「逾 5.2 万次有效调用」")
            continue
        bad("数", f"缺少关键数字 {k}（{label}）")

# ---------- 6. 与源数据文件比对（能量化的部分） ----------
import json

jvr = ROOT / "runs" / "c4-5arm-nothink" / "judge_vs_rule.json"
if jvr.exists():
    d = json.loads(jvr.read_text(encoding="utf-8"))
    arms = d.get("arms", d)
    for name, v in arms.items():
        b = v.get("binary", {})
        note(
            "判",
            f"{name}: 覆盖率={b.get('coverage')} 一致率={b.get('accuracy')} kappa={b.get('cohen_kappa')}",
        )

# ---------- 输出 ----------
print("=" * 72)
print(f"报告：{REP.name}  ({len(text)} 字符 / {len(lines)} 行)")
print("=" * 72)
if problems:
    print(f"\n发现 {len(problems)} 处问题：\n")
    for p in problems:
        print("  X", p)
else:
    print("\n未发现结构性问题。")
if notes:
    print(f"\n附注 {len(notes)} 条：\n")
    for n in notes:
        print("  -", n)
sys.exit(1 if problems else 0)
