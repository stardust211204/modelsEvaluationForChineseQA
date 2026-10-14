"""为 §3.5 的比值表重算「逐对不一致率」口径。

为什么要换
    现在表里的比值 = 选项重排翻转率（C6，4 次观测）÷ 重复作答翻转率（C3，10 次观测）。
    但翻转率**随观测次数单调上升**，两侧次数不同 ⇒ 分母被抬高、比值被系统性低估。
    项目自己的 analyze_position.py 已写明这一点，并把「逐对不一致率」定为主口径。

本脚本
    · 逐对不一致率 = 任取两次观测、答案不同的概率（与观测次数无关）
    · C6 的答案先**还原到原题坐标系**；None 跳过
    · 同时给出 MMLU-ProX（93 题）与 C-Eval（100 题）
"""

import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from meval import build_set, exclusions, paths  # noqa: E402
from meval.types import ParseMode  # noqa: E402

STRICT = ParseMode.STRICT.value
DROP = exclusions.load()
ARMS = ["deepseek-v4-pro", "longcat-2.5-preview-free", "deepseek-v4.1-flash",
        "qwen3.8-flash", "mimo-v2.6-flash"]
# ★ 最后一个字段是 `permuted` —— **两个数据集都必须是 True**。
#   C-Eval 的 c6 同样是乱序的（rep 0 原序、rep 1/2/3 各一种随机排列，实测非恒等）。
#   若给 C-Eval 传 False，就会跨越坐标系比较，逐对不一致率虚高到 76% 左右、
#   比值虚高到 25~58 —— 数字看着还挺像回事，但完全是假的。
SETS = {
    "MMLU-ProX": ("c6-5arm-v2", "c3-5arm-v2", "eval_set.jsonl", True),
    "C-Eval": ("ceval-c6-5arm", "ceval-c3-5arm", "ceval_set.jsonl", True),
}


def load(run, cfg, setfile, permuted):
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / setfile)}
    g = defaultdict(dict)
    p = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    if not p.exists():
        return None, None
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != STRICT:
            continue
        idx = t.get("answer_index")
        if permuted and t["instance_id"] in pool:
            _, perm = build_set.permute_instance(pool[t["instance_id"]],
                                                 t.get("rep", 0))
            idx = build_set.unpermute_answer(idx, perm)
        g[(t["arm"], t["instance_id"])][t.get("rep", 0)] = idx
    return g, pool


def pairwise(g, arm, items, reps):
    eq = n = 0
    for i in items:
        vals = [g[(arm, i)].get(r) for r in reps]
        vals = [v for v in vals if v is not None]
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                n += 1
                eq += int(vals[a] == vals[b])
    return (1 - eq / n) * 100 if n else float("nan")


for dsname, (c6run, c3run, setfile, perm) in SETS.items():
    C6, _ = load(c6run, "c6", setfile, perm)
    C3, _ = load(c3run, "c3", setfile, False)
    if C6 is None or C3 is None:
        print(f"  {dsname}: run 缺失")
        continue
    drop = DROP if dsname == "MMLU-ProX" else set()
    print("=" * 94)
    print(f"  {dsname}   逐对不一致率口径")
    print("=" * 94)
    print(f"  {'模型':<28}{'选项顺序逐对':>13}{'重复作答逐对':>13}{'比值':>8}")
    for arm in ARMS:
        items = sorted({i for (a, i) in C6 if a == arm and i not in drop})
        o = pairwise(C6, arm, items, [1, 2, 3])
        r = pairwise(C3, arm, items, list(range(10)))
        print(f"  {arm:<28}{o:>12.1f}%{r:>12.1f}%{o/r:>8.2f}")
    print()
