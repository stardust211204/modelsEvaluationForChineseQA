# -*- coding: utf-8 -*-
"""逐格复算 report_external_helm_polished.md 里的表格数字。

只做「报告单元格 vs 从 runs/ 重算」的比对，不改任何数据。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meval import build_set, exclusions, metrics, paths  # noqa: E402

ARMS = [
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
    "longcat-2.5-preview-free",
    "mimo-v2.6-flash",
    "qwen3.8-flash",
]
SH = {
    "deepseek-v4.1-flash": "ds-flash",
    "deepseek-v4-pro": "ds-pro",
    "longcat-2.5-preview-free": "longcat",
    "mimo-v2.6-flash": "mimo",
    "qwen3.8-flash": "qwen",
}
DROP = exclusions.load()

_CACHE = {}


def tallies(rid, cfg):
    key = (rid, cfg)
    if key not in _CACHE:
        d = next(
            (b / rid for b in (paths.ROOT / "runs", paths.ROOT / "archive" / "runs") if (b / rid).exists()),
            None,
        )
        f = d / "tallies" / cfg / "all.jsonl"
        ts = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        _CACHE[key] = [t for t in ts if t["instance_id"] not in DROP]
    return _CACHE[key]


def acc(rid, cfg, arm, mode="strict"):
    ts = [t for t in tallies(rid, cfg) if t["arm"] == arm and t["mode"] == mode]
    c = sum(1 for t in ts if t["tier"] == "correct")
    w = sum(1 for t in ts if t["tier"] == "wrong")
    return (c / (c + w) * 100 if c + w else float("nan")), c, w, len(ts)


print("=" * 100)
print(" §3.1 准确率表（报告 15 格）vs 重算")
print("=" * 100)
REPORT = {
    "longcat-2.5-preview-free": (75.8, 62.9, 88.0),
    "deepseek-v4.1-flash": (74.7, 64.1, 84.6),
    "qwen3.8-flash": (74.0, 61.2, 85.9),
    "deepseek-v4-pro": (68.8, 58.6, 78.4),
    "mimo-v2.6-flash": (63.0, 49.1, 75.9),
}
for mode in ("strict", "lenient"):
    print(f"\n--- {mode} 口径 ---")
    print(f"  {'臂':<10}{'MMLU':>9}{'C-Eval':>9}{'合并':>9}   {'报告(合并/MMLU/C-Eval)':>26}  判定")
    print("  " + "-" * 92)
    for arm in ARMS:
        am, cm, wm, tm = acc("c1-5arm-v2", "c1", arm, mode)
        ac, cc, wc, tc = acc("ceval-c1-5arm-v2", "c1", arm, mode)
        merged = (cm + cc) / (cm + wm + cc + wc) * 100
        r = REPORT[arm]
        ok = all(abs(v - x) < 0.05 for v, x in zip((merged, am, ac), r))
        flag = "OK " if ok else "XX "
        print(
            f"  {SH[arm]:<10}{am:>8.1f}%{ac:>8.1f}%{merged:>8.1f}%   "
            f"{r[0]:>7.1f}/{r[1]:>6.1f}/{r[2]:>6.1f}   {flag}"
            f" ({cm}/{cm+wm} + {cc}/{cc+wc})"
        )

print()
print("=" * 100)
print(" §2.4 T6 两口径可判条目（e6 四组，150 题 × 5 臂）")
print("=" * 100)
E6 = {
    "只答一个字母(A)": ("e6-a-5arm", "c2"),
    "可写理由(B)": ("e6-b-5arm", "c2"),
    "可写理由+末行(C)": ("e6-c-5arm", "c2"),
    "再加逐步推理(D)": ("e6-d-5arm", "c2"),
}
rep6 = {"只答一个字母(A)": (679, 680), "可写理由(B)": (0, 276),
        "可写理由+末行(C)": (0, 635), "再加逐步推理(D)": (0, 612)}
print(f"  {'组':<20}{'严格':>8}{'宽松':>8}   {'报告严格/宽松':>16}  判定")
for label, (rid, cfg) in E6.items():
    ts = tallies(rid, cfg)
    n_s = sum(1 for t in ts if t["mode"] == "strict" and t["tier"] in ("correct", "wrong"))
    n_l = sum(1 for t in ts if t["mode"] == "lenient" and t["tier"] in ("correct", "wrong"))
    rs, rl = rep6[label]
    flag = "OK " if (n_s == rs and n_l == rl) else "XX "
    print(f"  {label:<20}{n_s:>8}{n_l:>8}   {rs:>7}/{rl:<7}  {flag}  (总条目 {len(ts)//2})")

print()
print("=" * 100)
print(" §3.2 T16 稳定性 / 选项重排翻转率")
print("=" * 100)
rep16 = {
    "longcat-2.5-preview-free": (87.0, 97.0, 45.2, 18.0),
    "deepseek-v4-pro": (80.6, 93.0, 66.7, 45.0),
    "deepseek-v4.1-flash": (77.4, 83.0, 50.5, 21.0),
    "qwen3.8-flash": (69.9, 86.0, 57.0, 19.0),
    "mimo-v2.6-flash": (28.0, 55.0, 72.0, 38.0),
}


def stability(rid, cfg, arm):
    ts = [t for t in tallies(rid, cfg) if t["arm"] == arm and t["mode"] == "strict"]
    by = {}
    for t in ts:
        by.setdefault(t["instance_id"], []).append(t["answer_index"])
    same = sum(1 for v in by.values() if len(v) == len(set(v)) and None not in v)
    return same / len(by) * 100, len(by)


def flip(rid, cfg, arm, permuted):
    """翻转率：还原坐标系后，观测中出现 ≥2 个不同取值的题占比（None 跳过）。"""
    ts = [t for t in tallies(rid, cfg) if t["arm"] == arm and t["mode"] == "strict"]
    by = {}
    for t in ts:
        by.setdefault(t["instance_id"], []).append((t.get("perm"), t["answer_index"], t["tier"]))
    n_flip = 0
    for iid, obs in by.items():
        vals = set()
        for p, ai, tier in obs:
            if ai is None:
                continue
            vals.add(ai if not permuted else ai)
        if len(vals) >= 2:
            n_flip += 1
    return n_flip / len(by) * 100, len(by)


print("  （翻转率需要还原坐标系，见 _ratio_pairwise.py；此处仅列稳定性）")
print(f"  {'臂':<10}{'MMLU稳定性':>12}{'n':>5}{'C-Eval稳定性':>14}{'n':>5}   报告(MMLU/C-Eval)")
for arm in ARMS:
    s1, n1 = stability("c3-5arm-v2", "c3", arm)
    s2, n2 = stability("ceval-c3-5arm", "c3", arm)
    r = rep16[arm]
    f1 = "OK" if abs(s1 - r[0]) < 0.05 else "XX"
    f2 = "OK" if abs(s2 - r[1]) < 0.05 else "XX"
    print(f"  {SH[arm]:<10}{s1:>11.1f}%{n1:>5}{s2:>13.1f}%{n2:>5}   {r[0]:>5.1f}/{r[1]:<5.1f}  {f1} {f2}")

print()
print("=" * 100)
print(" ★ 稳定性排序是否「两集完全一致」")
print("=" * 100)
o1 = sorted(ARMS, key=lambda a: -stability("c3-5arm-v2", "c3", a)[0])
o2 = sorted(ARMS, key=lambda a: -stability("ceval-c3-5arm", "c3", a)[0])
print("  MMLU-ProX 序：", " > ".join(SH[a] for a in o1))
print("  C-Eval    序：", " > ".join(SH[a] for a in o2))
print("  →", "完全一致" if o1 == o2 else "★★ 不一致：报告称「完全一致」不成立")
