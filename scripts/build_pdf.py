r"""把 Markdown 报告转成 PDF（pandoc + xelatex + Eisvogel 模板）。

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

★ 三个版式约定（都在本脚本里处理，markdown 不用管）
    1. **章节编号只有一个来源**。标题里已手写编号（`# 1 评测概述`）的报告，
       不再传 `--number-sections`，否则会渲染成「1 1 评测概述」。
       用 `--numbering on|off` 可强制覆盖。
    2. **目录与正文分页**。`toc-own-page` 让正文另起一页；
       目录**与摘要同页**（封面 → 摘要 + 目录 → 正文）。不要改成目录单独一页 ——
       摘要只有几行，单独占一页会显得很空。
    3. **中文段落格式**：宋体正文 + Times New Roman 西文 + 单倍行距 + 首行缩进 1em。
       这套参数取自 CCL 2026 官方中文论文模板（调研产物在 `reports/_verify/web/`）。
    4. **分级标题**：一级 12pt 黑体、二三级 10pt 黑体（同样取自 CCL 模板）。
       实现在 `latex_headings.tex` —— 注意 **xeCJK 没有 `\CJKsffamily`**
       （那是 ctex 的接口），要单独切中文必须 `\setCJKfamilyfont` + `\CJKfamily`。

★ 字体为什么要分三个参数传
    以前 `mainfont` 同时兼作 `CJKmainfont`（都用微软雅黑），中文靠它兜底。
    西文换成 Times New Roman 之后，**中文字体必须单独指定** ——
    否则中文会落到 Times 上，而 Times 没有汉字字形，整篇中文会**静默渲染成空白**
    （xelatex 只在 log 里报 `Missing character`，不报错）。
    同理 `CJKmonofont` 必须设：行内代码走 monofont（Consolas），
    Consolas 没有汉字字形，代码里的中文会变空白。

用法
    .venv\\Scripts\\python.exe scripts\\build_pdf.py
    .venv\\Scripts\\python.exe scripts\\build_pdf.py --src reports/_archive/report_external_helm_full.md
    .venv\\Scripts\\python.exe scripts\\build_pdf.py --author "某某" --date "2026-10-05"

默认源是精简版 `reports/report_external_helm_polished.md` —— 它自 2026-10-17 起是
**唯一权威对外报告**；完整版已冻结归档到 `reports/_archive/`，需要时用 --src 显式指定。
"""

from __future__ import annotations

import argparse
import os
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

# ★ 封面背景：路径必须**相对于 pandoc 的执行目录**，--resource-path 对它无效。
#   所以这里存的是相对路径，且文件放在被 git 跟踪的 reports/assets/ 下
#   （Eisvogel 自带的 examples/ 整个目录被 .gitignore 忽略，不能直接引用）。
COVER_BACKGROUND = "reports/assets/cover_background.pdf"
# 内页背景：与封面同一张几何图形，但**预先调淡**（保留 20% 原色）。
# ★ 不能靠 Eisvogel 的 page-background-opacity —— background 宏包的 opacity
#   对嵌入的 PDF **不生效**：实测传 0.2 后渲染像素与封面完全同色（#E10058），
#   一点没淡。所以改用 scripts/make_page_background.py 预先生成淡版。
PAGE_BACKGROUND = "reports/assets/page_background.pdf"
# ★ 必须显式传 1.0。模板在不传时的默认值是 0.2，会**再淡一层**
#   （实测：预调淡 20% × 模板 0.2 = 实际只剩 4%，几乎看不见）。
#   淡的程度完全由 make_page_background.py 的参数控制，模板这层关掉。
PAGE_BACKGROUND_OPACITY = "1.0"

VSPACE = [
    # ★ 必须设 lang=zh-CN，否则图注前缀是英文 "Figure"；设了才是「图」
    "-V", "lang=zh-CN",
    "-V", "fontsize=10pt",
    # 行距由 --linestretch 传，**不写死在这里**。定为 1.5 倍 —— 中文报告的常规行距，
    # 单倍在 10pt 宋体正文下偏挤。
    # ★ 改行距会改页数（单倍 28 页 → 1.5 倍 35 页），凡登记了页数的地方都要同步；
    #   交付面上只有 .gitignore 的说明里写了页数。
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


def count_embedded_images(pdf: Path) -> int:
    """数 PDF 里**真正内嵌**的图片对象数。返回 -1 表示没有 pdfimages 可用。

    为什么需要它：`count_captions` 数的是 `图 N:` 这段**文字**，
    而 pandoc 找不到图片文件时**照样输出图注** —— 于是「图注 14/14」可以在
    「14 张图一张都没进去」的情况下成立。这个函数才是那段检查的补集。
    """
    if not shutil.which("pdfimages"):
        return -1
    r = subprocess.run(["pdfimages", "-list", str(pdf)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 输出前两行是表头（页码行 + 列名行）
    return max(0, len([ln for ln in (r.stdout or "").splitlines()[2:] if ln.strip()]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown 报告 -> PDF（Eisvogel）")
    ap.add_argument("--src", default="reports/report_external_helm_polished.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mainfont", default="Times New Roman",
                    help="西文正文字体")
    ap.add_argument("--monofont", default="Consolas", help="西文等宽（行内代码）")
    ap.add_argument("--cjkfont", default="SimSun", help="中文正文（宋体）")
    ap.add_argument("--cjksansfont", default="SimHei",
                    help="中文无衬线（黑体）。CJK 的粗体没有独立字面时会回退到它")
    ap.add_argument("--cjkmonofont", default="SimSun",
                    help="行内代码里的中文。**必须设**，否则该处中文渲染成空白")
    ap.add_argument("--linestretch", default="1.5",
                    help="行距倍率。默认 1.5 —— 中文报告常规行距，10pt 宋体正文下比单倍好读")
    ap.add_argument("--sansfont", default="SimHei",
                    help="西文无衬线（\\sffamily 的拉丁部分）。Eisvogel 封面用 "
                         "\\textsf 排 subtitle 与 date，不显式给就落回 LaTeX 默认的 "
                         "Latin Modern Sans。默认 SimHei，与封面中文标题同族")
    ap.add_argument("--author", default="", help="封面与页脚作者；留空则不显示")
    ap.add_argument("--date", default="", help="封面日期；留空则用 pandoc 默认")
    ap.add_argument("--cover-background", default=COVER_BACKGROUND,
                    help="封面背景 PDF（相对仓库根）；传空字符串则用无背景封面")
    ap.add_argument("--page-background", default=PAGE_BACKGROUND,
                    help="内页背景 PDF（相对仓库根）；传空字符串则用无背景内页")
    ap.add_argument("--page-background-opacity", default=PAGE_BACKGROUND_OPACITY,
                    help="内页背景不透明度；满不透明度会压掉正文可读性")
    ap.add_argument("--no-toc", action="store_true")
    ap.add_argument("--numbering", choices=("auto", "on", "off"), default="auto",
                    help="章节编号来源。auto（默认）：标题里已手写编号就不再加自动编号；"
                         "on：强制 LaTeX 自动编号；off：强制沿用标题里的编号")
    args = ap.parse_args()

    src = paths.ROOT / args.src
    if not src.exists():
        print(f"  源文件不存在: {src}")
        return 1
    if not TEMPLATE.exists():
        print(f"  [失败] 找不到模板 {TEMPLATE.relative_to(paths.ROOT)}")
        # ★ 2026-10-18 起模板本体**已入库**（reports/Eisvogel-3.5.1/eisvogel.latex，
        #   29 KB）。原来整个目录被 .gitignore 忽略，导致克隆仓库的人重建 PDF
        #   直接失败 —— 封面、字号、字体、标题样式、页眉页脚、配色全都拿不到。
        #   所以这句提示从「需重新放置 30 MB 的 Eisvogel」改成「从仓库取回」。
        print("         该文件应随仓库提供。若缺失，请执行：")
        print("           git checkout -- reports/Eisvogel-3.5.1/")
        print("        整包（含 examples/、30 MB）仍被 .gitignore 忽略，本项目不需要它。")
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

    # ★ 章节编号只能有一个来源。
    #   完整版（现归档于 _archive/report_external_helm_full.md）的标题不带编号，
    #   靠 LaTeX 自动编号；精简版（现为唯一权威报告）的标题**自带**编号（`# 1 评测概述`）。
    #   两种都传 --number-sections 会渲染成「1 1 评测概述」。这里按标题自动判定。
    h1 = re.search(r"^#\s+(.+)$", md, re.M)
    manual_num = bool(h1 and re.match(r"^\d+(?:\.\d+)*[\s.、]", h1.group(1)))
    if args.numbering == "auto":
        number_sections = not manual_num
    else:
        number_sections = args.numbering == "on"

    cmd = [
        "pandoc", str(src), "-o", str(out),
        "--template", str(TEMPLATE),
        "--pdf-engine=xelatex",
        # ★ 图片路径要同时满足两个读者，所以 resource-path 必须给两条：
        #   - **GitHub 预览**按文件所在目录解析，所以 markdown 里写 `../figures/x.png`
        #     （写 `figures/x.png` 在仓库根跑得通，但 GitHub 上 14 张图全是裂的）；
        #   - **pandoc** 从仓库根跑，靠 resource-path 把 `../figures/x.png` 折回来 ——
        #     给 <ROOT> 与 <源文件目录> 两条，后者拼出 `reports/../figures/x.png` 即命中。
        "--resource-path", os.pathsep.join([str(paths.ROOT), str(src.parent)]),
        "-V", f"mainfont={args.mainfont}",
        "-V", f"monofont={args.monofont}",
        # ★ sansfont 必须显式给。Eisvogel 的封面用 \textsf 排 subtitle 与 date
        #   （eisvogel.latex:935 `{\Large \textsf{$subtitle$}}`），而 pandoc
        #   只在给了 -V sansfont 时才写 \setsansfont。不给的话 \sffamily 落回
        #   LaTeX 默认的 **Latin Modern Sans** —— 封面日期就是这样变成了
        #   `LMSans12`，一套跟我们选的字体毫无关系的系统自带字体。
        #   实测（2026-10-18，pdftohtml -xml 逐段取字体）：
        #     标题 SimHei(30) / 副标题 SimHei(21) / **日期 LMSans12** ← 只它不对
        #   中文不受影响：xeCJK 下 \sffamily 的汉字走 CJKsansfont(SimHei)，本就对。
        "-V", f"sansfont={args.sansfont}",
        # ★ 中文字体分三档传。以前 mainfont 同时兼作 CJKmainfont（都用雅黑），
        #   现在西文换成 Times New Roman，就必须单独指定中文，
        #   否则中文会落到 Times 上 → 整篇缺字。
        "-V", f"CJKmainfont={args.cjkfont}",
        "-V", f"CJKsansfont={args.cjksansfont}",
        "-V", f"CJKmonofont={args.cjkmonofont}",
        # ★ 首行缩进：Eisvogel 默认走 `\KOMAoptions{parskip=half}`（首行不缩进、
        #   段间空行）。传 indent=true 会让它**跳过**那段设置，改由
        #   latex_indentfirst.tex 显式定成 1em。
        "-V", "indent=true",
        "-H", str(_SCRIPTS / "latex_indentfirst.tex"),
        # ★ 分级标题：一级 12pt 黑体、二三级 10pt 黑体（依据 CCL 2026 中文模板）
        "-H", str(_SCRIPTS / "latex_headings.tex"),
        "-V", f"linestretch={args.linestretch}",
        "-V", "titlepage=true",
        "-V", f"title={pick_title(md)}",
        # ★ 章节编号交给 LaTeX，但**仅在报告标题没手写编号时**。
        #   手写了再传这个参数会变成「1 1 评测概述」，见上面的 manual_num 判定。
        *(["--number-sections"] if number_sections else []),
        *VSPACE,
    ]
    bg = Path(args.cover_background)
    if args.cover_background and (paths.ROOT / bg).exists():
        cmd += ["-V", f"titlepage-background={bg.as_posix()}"]
    elif args.cover_background:
        print(f"  [警告] 封面背景不存在，改用无背景封面：{args.cover_background}")

    pbg = Path(args.page_background)
    if args.page_background and (paths.ROOT / pbg).exists():
        cmd += ["-V", f"page-background={pbg.as_posix()}"]
        if args.page_background_opacity:
            cmd += ["-V", f"page-background-opacity={args.page_background_opacity}"]
        # ★ 第 1 页不画内页背景：封面自带 titlepage-background，两者同时生效
        #   会在该页留下悬空的 ExtGState 引用。详见 scripts/latex_pagebg.tex。
        cmd += ["-H", str(_SCRIPTS / "latex_pagebg.tex")]
    elif args.page_background:
        print(f"  [警告] 内页背景不存在，改用无背景内页：{args.page_background}")
    if args.author:
        cmd += ["-V", f"author={args.author}"]
    if args.date:
        cmd += ["-V", f"date={args.date}"]
    if not args.no_toc:
        cmd += ["--toc", "-V", "toc-title=目录", "--toc-depth=2",
                # ★ 目录与正文分页（但**与摘要同页**）：
                #   Eisvogel 的 toc-own-page 在 \tableofcontents 之后插 \newpage，
                #   正文因此另起一页；目录本身不另起页，紧跟摘要。
                "-V", "toc-own-page=true"]

    print("=" * 78)
    print("  构建 PDF（Eisvogel）")
    print("=" * 78)
    print(f"  源   : {src.relative_to(paths.ROOT)}  ({len(md):,} 字符)")
    print(f"  编号 : {'LaTeX 自动编号（标题未手写）' if number_sections else '沿用标题里的手写编号'}"
          f"{'  [--numbering 强制]' if args.numbering != 'auto' else ''}")
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
    try:
        shown = out.relative_to(paths.ROOT)
    except ValueError:                     # --out 指到了仓库外
        shown = out
    print(f"  产出 : {shown}  ({out.stat().st_size / 1024:,.0f} KB)")
    print(f"  图注 : {n_fig} / {n_img}{flag}")

    # ★ 图注数**证明不了图片被嵌进去了**（2026-10-17 实测）：
    #   pandoc 找不到图片文件时只打一行 warning，**图注照输出** ——
    #   实测把路径写错一层后，count_captions 仍报 14/14，而 PDF 从 1,329 KB 掉到 407 KB、
    #   内嵌图片从 52 掉到 20。也就是说这道检查对「图全没了」完全无感。
    #   所以另加两道：
    #   ① 构建前式的**路径可解析性**检查：图片路径必须能相对**源文件所在目录**解析 ——
    #      这正是 GitHub 的解析规则。写 `figures/x.png` 时 pandoc 从仓库根跑得通，
    #      但 GitHub 上整页裂图（本项目踩过）。
    #   ② 内嵌图片数（pdfimages）与文件体积，作为「图确实进去了」的旁证。
    src_dir = src.parent
    unresolved = [p for p in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", md)
                  if not p.startswith(("http", "/")) and not (src_dir / p).exists()]
    if unresolved:
        print(f"  ★ 图片路径 {len(unresolved)} 处无法相对源文件目录解析（pandoc 仍可能靠"
              f" resource-path 找到，但 **GitHub 预览会裂图**）：")
        for p in dict.fromkeys(unresolved):
            print(f"      {p}")
    n_embed = count_embedded_images(out)
    if n_embed >= 0:
        # 正常时应 ≳ 图片引用数（Eisvogel 封面底图与部分 PNG 会拆成多个对象）
        warn = "" if n_embed >= n_img else f"   ★ 只有 {n_embed} 个，疑似图片未嵌入"
        print(f"  内嵌图: {n_embed} 个（来源 {n_img} 张）{warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
