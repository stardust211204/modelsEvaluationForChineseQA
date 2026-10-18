"""E5 5-shot 受控重做与数量梯度分析。

设计
    示例梯度**锚定在既有 c5b 的 5 条上**并逐档**嵌套**（n=1 ⊂ n=3 ⊂ n=5 ⊂ n=10），
    相邻档之间只差新增的那几条 ⇒ 单因子。示例的选项数贴近目标题（受控）。

    ★ 0-shot 基线**必须按数据集各跑一个**：
      `e6-a-5arm`（= `--config c2 --template c1_strict.txt`）跑的是 **eval_set**，
      它的 tally 里全是 `prox-zh-*`。拿它当 C-Eval 的基线会得到
      「与 MMLU-ProX 完全相同的 0-shot 数字」+「配对交集为空 ⇒ p=nan」——
      这正是本脚本第一版踩的坑。C-Eval 侧因此单跑了 `e5-ceval-n0-5arm`。

    ★ 0-shot 与 c5 同题集：c2 与 c5 都是 `sampling: from_c1, n_instances: 150`
      ⇒ `subset_from_c1(pool, 150)` 是确定性函数 ⇒ 落在同一批 150 题。

    ★ `c5b` 与 E5 的 n=5 **逐条同示例**、仅跑批时段不同
      ⇒ 两者之差是**天然的时段对照**，可直接量化混淆量级。
"""

import json
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import build_set, exclusions, metrics, paths
from meval.types import Config, ParseMode

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
DROP = exclusions.load()

# dataset -> (题集文件, 各档 run_id)
# ★ 实测：C5 族只有 3 个 run —— c5-5arm-v3(eval) / ceval-c5-5arm-v3 / ceval-c5b-5arm。
#   **MMLU-ProX 侧没有 c5b 跑批** ⇒ 那边的 n=5 无历史对照（它本身就是首个
#   c5b 式 eval_set 跑批），时段对照只在 C-Eval 侧成立。
PLAN = {
    "MMLU-ProX": ("eval_set.jsonl", {
        0: ("e6-a-5arm", "c2"),          # 0-shot 基线（同期新跑）
        1: ("e5-mmlu-n1-5arm", "c5"), 3: ("e5-mmlu-n3-5arm", "c5"),
        5: ("e5-mmlu-n5-5arm", "c5"), 10: ("e5-mmlu-n10-5arm", "c5"),
    }),
    "C-Eval": ("ceval_set.jsonl", {
        0: ("e5-ceval-n0-5arm", "c2"),   # ★ 必须**单独跑** —— 见下方说明
        1: ("e5-ceval-n1-5arm", "c5"), 3: ("e5-ceval-n3-5arm", "c5"),
        5: ("e5-ceval-n5-5arm", "c5"), 10: ("e5-ceval-n10-5arm", "c5"),
        "c5b对照": ("ceval-c5b-5arm", "c5b"),   # 同示例、旧时段
    }),
}
ORIG_C5 = {"MMLU-ProX": ("c5-5arm-v3", "c5"), "C-Eval": ("ceval-c5-5arm-v3", "c5")}


def load(run, cfg):
    g = {}
    p = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    if not p.exists():
        return g
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        if t["mode"] != ParseMode.STRICT.value or t["instance_id"] in DROP:
            continue
        if t["arm"] not in ARMS:
            continue
        g[(t["arm"], t["instance_id"])] = t["tier"]
    return g


def acc(t, arm):
    c = sum(1 for (a, i), v in t.items() if a == arm and v == "correct")
    w = sum(1 for (a, i), v in t.items() if a == arm and v == "wrong")
    return (c / (c + w) * 100 if c + w else float("nan")), c + w


print("=" * 108)
print("  E5 5-shot 受控重做与数量梯度")
print("=" * 108)

for ds, (setfile, levels) in PLAN.items():
    T = {k: load(r, c) for k, (r, c) in levels.items()}
    have = {k: v for k, v in T.items() if v}
    if not have:
        print(f"\n  ▸ {ds}: 尚无 tally，先跑批")
        continue
    print(f"\n{'='*108}\n  ▸ {ds}\n{'='*108}")

    # ---- 表1：准确率曲线 ---------------------------------------------------
    ks = sorted(k for k in have if isinstance(k, int))
    print(f"\n  【表1】准确率随示例数量（严格口径）")
    print(f"  {'模型':<10}" + "".join(f"{'n='+str(k):>10}" for k in ks)
          + f"{'n=0→10':>10}{'n=0→5':>9}")
    for arm in ARMS:
        row = [acc(have[k], arm)[0] for k in ks]
        # ★ a0 必须真取 n=0；缺基线时 `row[0]` 是 n=1，会把标签算错
        a0 = acc(have[0], arm)[0] if 0 in have else float("nan")
        a10 = acc(have[10], arm)[0] if 10 in have else float("nan")
        a5 = acc(have[5], arm)[0] if 5 in have else float("nan")
        print(f"  {SH[arm]:<10}" + "".join(f"{x:>9.1f}%" for x in row)
              + f"{a10-a0:>+9.1f}{a5-a0:>+8.1f}")

    # ---- 表2：配对 McNemar（0-shot vs 各档）--------------------------------
    print(f"\n  【表2】0-shot vs 各档的配对 McNemar（同 150 题）")
    if 0 not in have:
        print("      （缺 0-shot 基线）")
    else:
        for k in ks[1:]:
            print(f"\n    ▸ 0-shot → n={k}")
            print(f"      {'模型':<10}{'n=0':>8}{'n='+str(k):>8}{'差':>8}{'b':>5}{'c':>5}{'p':>11}")
            tb = tc = 0
            ds_ = []
            for arm in ARMS:
                a0, _ = acc(have[0], arm)
                a1, _ = acc(have[k], arm)
                b = c = 0
                for iid in {i for (a, i) in have[0] if a == arm} & \
                           {i for (a, i) in have[k] if a == arm}:
                    x, y = have[0][(arm, iid)], have[k][(arm, iid)]
                    if x not in ("correct", "wrong") or y not in ("correct", "wrong"):
                        continue
                    if x == "correct" and y == "wrong":
                        b += 1
                    elif x == "wrong" and y == "correct":
                        c += 1
                tb += b
                tc += c
                ds_.append(a1 - a0)
                p = metrics.mcnemar_exact(b, c) if b + c else float("nan")
                print(f"      {SH[arm]:<10}{a0:>7.1f}%{a1:>7.1f}%{a1-a0:>+7.1f}"
                      f"{b:>5}{c:>5}{p:>11.4f}")
            p = metrics.mcnemar_exact(tb, tc) if tb + tc else float("nan")
            print(f"      {'合并五个模型':<10}{'':>8}{'':>8}{statistics.mean(ds_):>+7.1f}"
                  f"{tb:>5}{tc:>5}{p:>11.4g}")

    # ---- 表3：c5b 时段对照 -------------------------------------------------
    if "c5b对照" in have and 5 in have:
        print(f"\n  【表3】★ 时段对照：E5 的 n=5 与既有 c5b **逐条同示例**，仅跑批时段不同")
        print(f"      {'模型':<10}{'E5 n=5（新）':>13}{'c5b（10-07）':>13}{'差':>9}")
        for arm in ARMS:
            a, _ = acc(have[5], arm)
            b, _ = acc(have["c5b对照"], arm)
            print(f"      {SH[arm]:<10}{a:>12.1f}%{b:>12.1f}%{a-b:>+8.1f}")
        print("      ⇒ 这个差就是「时段漂移」在本实验里的量级，读梯度时必须先扣掉它。")

    # ---- 表4：受控 vs 原始 C5 ---------------------------------------------
    print(f"\n  【表4】与原始 C5 的对照（原 C5 的示例选项数与目标题错配）")
    orun, ocfg = ORIG_C5[ds]
    orig = load(orun, ocfg)
    if orig and 0 in have:
        print(f"      {'模型':<10}{'0-shot':>9}{'原 C5':>9}{'受控 n=5':>10}"
              f"{'原C5−0':>9}{'受控−0':>9}")
        for arm in ARMS:
            a0, _ = acc(have[0], arm)
            ao, _ = acc(orig, arm)
            a5, _ = acc(have[5], arm) if 5 in have else (float("nan"), 0)
            print(f"      {SH[arm]:<10}{a0:>8.1f}%{ao:>8.1f}%{a5:>9.1f}%"
                  f"{ao-a0:>+8.1f}{a5-a0:>+8.1f}")
        print("      ⇒ 若受控后的增益显著大于原始 C5，则 §3.6 的崩塌确实来自格式错配。")
