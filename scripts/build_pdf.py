"""把 Markdown 报告转成 PDF（pandoc + xelatex + Eisvogel 模板）。

产物
    A4、带封面与页眉页脚、图注为中文编号（图 1、图 2…）的报告 PDF。

为什么用 Eisvogel
    早期用 pandoc 自带模板 + 自写 LaTeX 头，结果反复出现
    「图被页面下边缘裁掉、图注整条消失、xelatex 不报任何警告」。
    根因是底部浮动（`b`）：LaTeX 判断「放得下」的依据与实际排版高度不一致。
    Eisvogel 把 `\\floatplacement{figure}` 设成 **H**（行内放置，不是浮动），
    TeX 会像普通内容一样正常分页 —— 这个坑从根上没有了。
    用 `-V float-placement-figure=...` 可以改，但**不建议改**：
    实测 H 既不丢图注，图也留在各自小节旁边。

为什么用 xelatex 而不是别的
    中文 PDF 只有走 **xelatex / lualatex + 中文字体** 才可靠。
    wkhtmltopdf / weasyprint 路线在本机没装；pdfLaTeX 处理 CJK 需要额外宏包。

★ 三个会「静默丢内容」的坑（都实测踩过，构建后脚本会自检）
    1. **emoji 与部分符号在 Microsoft YaHei 里没有字形**（`⚠` `✅` `❌` `✔` `✖`）。
       xelatex 只打一行 warning，然后**渲染成空白**，页面不报错。
       ⇒ 报告正文约定：不用 emoji，要强调就用文字。
    2. **代码块与行内代码里不能有中文**。pandoc 用 `monofont`（Consolas）渲染它们，
       Consolas 没有中文字形 ⇒ 整段变空白。提示词之类的样例请用引用块（`>`）。
       独立校验：`scripts/_check_render_safe.py`
    3. **图注可能整条消失**（见上）。本脚本构建后会数一遍 `图 N:` 的条数，
       与 markdown 里 `![` 的个数比对，不一致会报警。
    * `—`(U+2014) `−`(U+2212) `→`(U+2192) `≤` `≥` `①②` 均正常，可放心用。

用法
    .venv\\Scripts\\python.exe scripts\\build_pdf.py
    .venv\\Scripts\\python.exe scripts\\build_pdf.py --src reports/report_external_helm.md
    .venv\\Scripts\\python.exe scripts\\build_pdf.py --author "某某" --date "2026-10-05"
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import paths  # noqa: E402

# 控制台是 GBK 时，打印缺字警告（含 emoji）会 UnicodeEncodeError 崩掉。
# 构建本身已经完成，但脚本会以非零码退出，看起来像构建失败 —— 必须修掉。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 实测在 Microsoft YaHei 中缺失、会被静默渲染成空白的字符
BAD_CHARS = {"\u26a0": "⚠ (U+26A0)", "\ufe0f": "变体选择符 (U+FE0F)",
             "\u2705": "✅ (U+2705)", "\u274c": "❌ (U+274C)",
             "\u2714": "✔ (U+2714)", "\u2716": "✖ (U+2716)"}

TEMPLATE = paths.ROOT / "reports" / "Eisvogel-3.5.1" / "eisvogel.latex"

VSPACE = [
    # ★ 必须设 lang=zh-CN，否则图注前缀是英文 "Figure"；设了才是「图」
    "-V", "lang=zh-CN",
    "-V", "fontsize=10pt",
    "-V", "linestretch=1.15",
    "-V", "colorlinks=true",
    "-V", "linkcolor=NavyBlue",
    "-V", "urlcolor=NavyBlue",
    "-V", "toccolor=black",
]


def check_chars(text: str) -> list[str]:
    """构建前查会静默丢字的字符。"""
    return [f"{name} x{text.count(ch)}" for ch, name in BAD_CHARS.items() if text.count(ch)]


def pick_title(md: str) -> str:
    """取封面标题。

    ★ 优先读 YAML front matter 的 `title:`。
      早先的写法是「取第一个一级标题」，在标题改为元数据驱动之后就取错了 ——
      正文第一个 `#` 是「评测概述」，于是封面印成了「评测概述」。
    """
    lines = md.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    for line in lines:                       # 回退：没有 front matter 时
        if line.startswith("# "):
            return line[2:].strip()
    return "评测报告"


def count_captions(pdf: Path) -> int:
    """数 PDF 里的图注条数。返回 -1 表示没有 pdftotext 可用。"""
    if not shutil.which("pdftotext"):
        return -1
    r = subprocess.run(["pdftotext", str(pdf), "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return len(re.findall(r"(?:Figure|图)\s*\d+\s*[:：]", r.stdout or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 报告 -> PDF（Eisvogel）")
    ap.add_argument("--src", default="reports/report_external_helm.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mainfont", default="Microsoft YaHei")
    ap.add_argument("--monofont", default="Consolas")
    ap.add_argument("--author", default="", help="封面与页脚作者；留空则不显示")
    ap.add_argument("--date", default="", help="封面日期；留空则用 pandoc 默认")
    ap.add_argument("--no-toc", action="store_true")
    args = ap.parse_args()

    src = paths.ROOT / args.src
    if not src.exists():
        print(f"  源文件不存在: {src}")
        return 1
    if not TEMPLATE.exists():
        print(f"  [失败] 找不到模板 {TEMPLATE.relative_to(paths.ROOT)}")
        print("         该目录被 .gitignore 忽略，换机器时需重新放置 Eisvogel 3.5.1。")
        return 1
    if not shutil.which("pandoc"):
        print("  pandoc 未安装")
        return 1

    out = Path(args.out) if args.out else src.with_suffix(".pdf")
    md = src.read_text(encoding="utf-8")

    hits = check_chars(md)
    if hits:
        print("  [警告] 报告里含以下字符，PDF 中会渲染成空白：")
        for h in hits:
            print(f"          {h}")
        print("         建议替换为文字表述。")

    cmd = [
        "pandoc", str(src), "-o", str(out),
        "--template", str(TEMPLATE),
        "--pdf-engine=xelatex",
        "--resource-path", str(paths.ROOT),
        "-V", f"mainfont={args.mainfont}",
        "-V", f"monofont={args.monofont}",
        "-V", f"CJKmainfont={args.mainfont}",
        "-V", "titlepage=true",
        "-V", f"title={pick_title(md)}",
        # ★ 章节编号交给 LaTeX。markdown 里**不要**再手写编号
        #   （写了会变成「1 1 评测概述」）。
        "--number-sections",
        *VSPACE,
    ]
    if args.author:
        cmd += ["-V", f"author={args.author}"]
    if args.date:
        cmd += ["-V", f"date={args.date}"]
    if not args.no_toc:
        cmd += ["--toc", "-V", "toc-title=目录", "--toc-depth=2"]

    print("=" * 78)
    print("  构建 PDF（Eisvogel）")
    print("=" * 78)
    print(f"  源   : {src.relative_to(paths.ROOT)}  ({len(md):,} 字符)")
    r = subprocess.run(cmd, cwd=paths.ROOT, capture_output=True, text=True, encoding="utf-8")

    warn = [l for l in (r.stderr or "").splitlines() if "Missing character" in l]
    other = [l for l in (r.stderr or "").splitlines() if l.strip() and "Missing character" not in l]
    if warn:
        print(f"  缺字警告 {len(warn)} 条（这些字符在 PDF 里是空白）：")
        for l in dict.fromkeys(warn):
            print(f"     {l.strip()[:110]}")
    for l in other[:6]:
        print(f"  {l.strip()[:110]}")

    if r.returncode != 0 or not out.exists():
        print(f"  [失败] pandoc 退出码 {r.returncode}")
        return 1

    # 构建后自检：图注会静默消失，必须数一遍
    n_fig, n_img = count_captions(out), md.count("![")
    flag = "" if n_fig == n_img else f"   ★ 少 {n_img - n_fig} 条 —— 检查浮动设置"
    print(f"  产出 : {out.relative_to(paths.ROOT)}  ({out.stat().st_size / 1024:,.0f} KB)")
    print(f"  图注 : {n_fig} / {n_img}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
