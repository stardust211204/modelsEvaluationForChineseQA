# -*- coding: utf-8 -*-
"""把 figures/latex/*.tex 的管线图编译成 figures/*.png。

用法
    .venv\\Scripts\\python.exe scripts\\build_pipeline_fig.py pipeline_main_v11 --out 40_pipeline_main
    .venv\\Scripts\\python.exe scripts\\build_pipeline_fig.py pipeline_main_v11 --preview   # 只看不覆盖

**默认不覆盖** figures/ 下的正式图件 —— 只有显式给 --out 才写过去。
`--preview` 只把 PNG 落在 figures/latex/ 里，供选版时对比。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "figures" / "latex"
FIGDIR = ROOT / "figures"
DPI = "150"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description="编译管线图 tex -> png")
    ap.add_argument("name", help="figures/latex/ 下的文件名（不含 .tex）")
    ap.add_argument("--out", default=None, help="写到 figures/<out>.png；不给则只出预览")
    ap.add_argument("--preview", action="store_true", help="只生成 figures/latex/ 下的预览 PNG")
    ap.add_argument("--dpi", default=DPI)
    args = ap.parse_args()

    tex = LATEX / f"{args.name}.tex"
    if not tex.exists():
        print(f"  源文件不存在: {tex}")
        return 1
    for exe in ("xelatex", "pdftoppm"):
        if not shutil.which(exe):
            print(f"  找不到 {exe}")
            return 1

    r = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"{args.name}.tex"], LATEX)
    pdf = LATEX / f"{args.name}.pdf"
    if r.returncode != 0 or not pdf.exists():
        print("  [失败] xelatex 编译不过：")
        for line in (r.stdout or "").splitlines():
            if line.startswith("!") or "Error" in line or "Undefined" in line:
                print(f"     {line.strip()[:120]}")
        return 1

    # pdftoppm 的输出名是 <prefix>-1.png
    r = run(["pdftoppm", "-png", "-r", args.dpi, "-f", "1", "-l", "1",
             f"{args.name}.pdf", f"{args.name}"], LATEX)
    png = LATEX / f"{args.name}-1.png"
    if not png.exists():
        print("  [失败] pdftoppm 没产出 PNG")
        print((r.stderr or "")[:400])
        return 1

    from PIL import Image  # noqa: PLC0415

    w, h = Image.open(png).size
    print(f"  编译成功: {args.name}.tex -> {png.name}  {w}x{h}px  宽高比 {w/h:.2f}")

    if args.out:
        dst = FIGDIR / f"{args.out}.png"
        shutil.copyfile(png, dst)
        print(f"  ★ 已写入正式图件: figures/{args.out}.png")
    elif not args.preview:
        print("  （未给 --out，仅生成预览）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
