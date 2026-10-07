"""把 Markdown 报告转成 PDF（pandoc + xelatex）。

为什么用 xelatex 而不是别的
    中文 PDF 只有走 **xelatex / lualatex + 中文字体** 才可靠。
    wkhtmltopdf / weasyprint 路线在本机没装；而且 pdfLaTeX 处理 CJK 需要额外宏包。

★ 已知的字体坑（实测踩过）
    * `⚠`(U+26A0) 与变体选择符 `️`(U+FE0F) **在 Microsoft YaHei 里没有**，
      xelatex 会打一行 warning 然后**渲染成空白** —— 页面不会报错，只是符号没了。
      ⇒ 本项目约定：报告正文**不使用 emoji**，需要强调就用文字。
    * `—`(U+2014) `−`(U+2212) `→`(U+2192) `≤` `≥` `①②` 均正常，可放心使用。

用法
    .venv\\Scripts\\python.exe scripts\\build_pdf.py
    .venv\\Scripts\\python.exe scripts\\build_pdf.py --src reports/report_internal.md
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

# 实测在 Microsoft YaHei 中缺失、会被静默渲染成空白的字符
BAD_CHARS = {"\u26a0": "⚠ (U+26A0)", "\ufe0f": "变体选择符 (U+FE0F)"}

VSPACE = [
    "-V", "geometry:margin=2.2cm",
    "-V", "fontsize=10pt",
    "-V", "linestretch=1.15",
    "-V", "colorlinks=true",
    "-V", "linkcolor=NavyBlue",
    "-V", "urlcolor=NavyBlue",
    "-V", "toccolor=black",
]


def check_chars(text: str) -> list[str]:
    hits: list[str] = []
    for ch, name in BAD_CHARS.items():
        n = text.count(ch)
        if n:
            hits.append(f"{name} x{n}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 报告 -> PDF")
    ap.add_argument("--src", default="reports/report_external.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mainfont", default="Microsoft YaHei")
    ap.add_argument("--monofont", default="Consolas")
    ap.add_argument("--toc", action="store_true", default=True)
    args = ap.parse_args()

    src = paths.ROOT / args.src
    if not src.exists():
        print(f"  源文件不存在: {src}")
        return 1
    out = Path(args.out) if args.out else src.with_suffix(".pdf")

    md = src.read_text(encoding="utf-8")

    # ---- 转换前先查会静默丢字的字符 -------------------------------------
    hits = check_chars(md)
    if hits:
        print("  [警告] 报告里含以下字符，PDF 中会渲染成空白：")
        for h in hits:
            print(f"          {h}")
        print("         建议替换为文字表述。")

    if not shutil.which("pandoc"):
        print("  pandoc 未安装")
        return 1

    cmd = [
        "pandoc", str(src), "-o", str(out),
        "--pdf-engine=xelatex",
        "-V", f"mainfont={args.mainfont}",
        "-V", f"monofont={args.monofont}",
        "--resource-path", str(paths.ROOT),
        *VSPACE,
    ]
    if args.toc:
        cmd.append("--toc")
        cmd += ["-V", "toc-title=目录", "--toc-depth=2"]

    print("=" * 78)
    print("  构建 PDF")
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
    print(f"  产出 : {out.relative_to(paths.ROOT)}  ({out.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
