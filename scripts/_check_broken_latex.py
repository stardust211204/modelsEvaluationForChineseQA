"""判定标准：`\\命令` 是否是**真实存在的 LaTeX 命令**。
不存在的 = 真损坏（丢空格/丢反斜杠）；存在的 = 只是"含公式"，不是缺陷。
"""

import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import build_set, exclusions, paths

# 真实 LaTeX / amsmath / amssymb / 常用包命令（用于判定"命令是否存在"）
REAL = set("""
frac dfrac tfrac cfrac sqrt sum prod int iint iiint oint lim log ln sin cos tan cot sec
csc arcsin arccos arctan sinh cosh tanh coth exp max min sup inf det dim ker deg gcd hom
mathrm mathbf mathit mathsf mathtt mathcal mathbb mathfrak mathnormal boldsymbol text
textbf textit textrm textsf texttt underline overline bar hat tilde vec dot ddot acute
grave check breve widetilde widehat overrightarrow overleftarrow overleftrightarrow
left right big Big bigg Bigg middle langle rangle lceil rceil lfloor rfloor vert Vert
begin end array matrix pmatrix bmatrix vmatrix Vmatrix cases aligned align gather split
quad qquad hspace vspace hline cline multicolumn hfill centerline linebreak newline
partial nabla infty pm mp times div cdot ast star circ bullet cap cup uplus sqcap sqcup
vee wedge setminus wr diamond bigtriangleup bigtriangledown triangleleft triangleright
oplus ominus otimes oslash odot bigcirc dagger ddagger amalg
leq geq neq equiv sim simeq approx cong propto prec succ preceq succeq ll gg subset
supset subseteq supseteq sqsubset sqsupset sqsubseteq sqsupseteq in ni notin parallel
perp mid nmid forall exists nexists emptyset varnothing angle triangle backslash prime
rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow mapsto longmapsto
longrightarrow longleftarrow longleftrightarrow uparrow downarrow updownarrow
rightharpoonup rightharpoondown leftharpoonup leftharpoondown rightleftharpoons
nearrow searrow swarrow nwarrow hookrightarrow hookleftarrow
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu
nu xi omicron pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
aleph hbar ell im wp Re Im prime partial infty clubsuit diamondsuit heartsuit spadesuit
dots cdots ldots vdots ddots mathellipsis dotsc dotsb dotsm dotsi dotso
limits nolimits displaystyle textstyle scriptstyle scriptscriptstyle
operatorname overbrace underbrace stackrel overset underset substack
because therefore leqslant geqslant nleqslant ngeqslant lnot lvert rvert lVert rVert
binom choose dbinom tbinom nicefrac raisebox rule phantom vphantom hphantom StrLeft
color textcolor mathcolor pagecolor boxed cancel
""".split())

TOK = re.compile(r"\\([a-zA-Z]+)")

CE = build_set.load_frozen(paths.data() / "ceval_set.jsonl")
EV = build_set.load_frozen(paths.data() / "eval_set.jsonl")
DROP = exclusions.load()


def txt(i):
    return i.question + " " + " ".join(i.options)


print("=" * 100)
print("  「不存在的 LaTeX 命令」= 真损坏")
print("=" * 100)
for pool, label in ((EV, "MMLU-ProX（588 题）"), (CE, "C-Eval（583 题）")):
    bad = defaultdict(set)
    allc = Counter()
    for i in pool:
        for m in TOK.finditer(txt(i)):
            c = m.group(1)
            allc[c] += 1
            if c not in REAL:
                bad[i.id].add("\\" + c)
    print(f"\n  ▸ {label}")
    print(f"     出现的不同命令 {len(allc)} 种")
    print(f"     命令不存在于 LaTeX 的题：{len(bad)}")
    ind = sum(1 for k in bad if k in DROP)
    print(f"       其中在排除清单里 {ind} / 不在 {len(bad)-ind}")
    for iid, cs in list(bad.items())[:10]:
        tag = "✓排除" if iid in DROP else "★未排除"
        print(f"       {tag}  {iid:<36}{sorted(cs)}")

print()
print("=" * 100)
print("  模式 3（字母后紧跟命令）与模式 4（空数学环境）的真实样例")
print("=" * 100)
P3 = exclusions.BAD_PATTERNS[2][0]
P4 = exclusions.BAD_PATTERNS[3][0]
for pool, label in ((CE, "C-Eval"), (EV, "MMLU-ProX")):
    print(f"\n  ▸ {label}")
    n = 0
    for i in pool:
        for m in re.finditer(P3, txt(i)):
            print(f"     [模式3] {i.id:<34} …{txt(i)[max(0,m.start()-28):m.end()+22]!r}")
            n += 1
            break
        if n >= 4:
            break
    n = 0
    for i in pool:
        for m in re.finditer(P4, txt(i)):
            print(f"     [模式4] {i.id:<34} …{txt(i)[max(0,m.start()-34):m.end()+30]!r}")
            n += 1
            break
        if n >= 4:
            break
