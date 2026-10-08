"""把封面背景调淡，生成内页用的背景。

为什么不能靠 Eisvogel 的 page-background-opacity
    模板用 `\\backgroundsetup{opacity=0.2}` 实现，但 background 宏包的 opacity
    对**嵌入的 PDF** 不生效 —— 实测渲染出来的像素与封面**完全同色**
    （#E10058 / #3DC5D5，一点没淡）。所以只能预先调淡。

做法
    把矢量背景按高分辨率栅格化，按权重与白色混合，再存成 PDF。
    颜色越接近 1 越淡（1.0 = 全白，看不见）。

用法
    .venv\\Scripts\\python.exe scripts\\make_page_background.py 0.2
    参数 = **原色保留比例**：0.2 表示只保留 20% 的原色（其余混白）。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "assets" / "cover_background.pdf"
DST = ROOT / "reports" / "assets" / "page_background.pdf"
DPI = 200


def main() -> int:
    keep = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    if not SRC.exists():
        print(f"!! 找不到源背景 {SRC}")
        return 1
    tmp = Path(tempfile.gettempdir()) / "_pbg"
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-f", "1", "-l", "1",
                    str(SRC), str(tmp)], check=True, capture_output=True)
    png = sorted(tmp.parent.glob(tmp.name + "*.png"))[0]
    im = Image.open(png).convert("RGB")
    white = Image.new("RGB", im.size, (255, 255, 255))
    # blend(white, im, keep)：keep=0.2 ⇒ 原色占 20%，其余是白
    out = Image.blend(white, im, keep)
    out.save(DST, "PDF", resolution=DPI)
    px = out.getpixel((int(im.width * 0.06), int(im.height * 0.9)))
    print(f"  源 {SRC.name}  {im.size} @ {DPI}dpi")
    print(f"  原色保留 {keep:.0%}（其余混白）")
    print(f"  品红区实测 #{px[0]:02X}{px[1]:02X}{px[2]:02X}（原色 #E10058）")
    print(f"  产 {DST.relative_to(ROOT)}  ({DST.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
