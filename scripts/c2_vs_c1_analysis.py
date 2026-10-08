"""C2（自由作答）vs C1（限制格式作答）的配对分析。

为什么要单独做这个分析
    报告 §3.7 只把「自由作答」当作**旗舰档对比的一个设置**，
    从未回答「去掉格式约束本身能带来多大提升」——
    而这恰恰是 C2 这个设置存在的唯一目的（见 matrix.yaml 的 factor: 格式约束）。

口径问题（关键）
    C1 的模型输出 99.9% 是裸字母 ⇒ 严格口径与宽松口径结果完全相同；
    C2 的模型被允许说明理由 ⇒ 严格口径命中率 0%（685 条全是 none），
    只能读宽松口径。
    ⇒ 因此必须用**同为宽松口径**来对比，才是干净的。
"""

import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from meval import build_set, exclusions, metrics, paths  # noqa: E402
from meval.types import Config  # noqa: E402

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
DROP = exclusions.load()

DATASETS = [
    ("MMLU-ProX", "eval_set.jsonl", ("c1-5arm-v2", "c1"), ("c2hi-5arm-v2", "c2")),
    ("C-Eval", "ceval_set.jsonl", ("ceval-c1-5arm-v2", "c1"), ("ceval-c2hi-5arm-v2", "c2")),
]


def tier_map(run, cfg, arm, mode):
    ts = metrics.load_tallies(run, Config(cfg))
    return {t["instance_id"]: t["tier"] for t in ts
            if t["arm"] == arm and t["mode"] == mode
            and t["instance_id"] not in DROP
            and t["tier"] in ("correct", "wrong")}


def main() -> int:
    for dsname, dsfile, base, free in DATASETS:
        pool = {i.id: i for i in build_set.load_frozen(paths.data() / dsfile)}
        print("═" * 80)
        print(f"  {dsname}：去掉格式约束（限制格式作答 → 自由作答）能提升多少？")
        print("═" * 80)
        print()
        print(f"  {'模型':<10}{'限制格式':>10}{'自由作答':>10}{'提升':>9}"
              f"{'变对':>7}{'变错':>7}{'p 值':>10}")
        print("  " + "-" * 70)

        details = []
        for a in ARMS:
            # ★ 两边都用宽松口径 —— C2 在严格口径下命中率为 0，不可比
            b = tier_map(*base, a, "lenient")
            f = tier_map(*free, a, "lenient")
            common = sorted(set(b) & set(f))
            if not common:
                continue
            accb = sum(1 for i in common if b[i] == "correct") / len(common) * 100
            accf = sum(1 for i in common if f[i] == "correct") / len(common) * 100
            up = sum(1 for i in common if f[i] == "correct" and b[i] != "correct")
            dn = sum(1 for i in common if b[i] == "correct" and f[i] != "correct")
            p = metrics.mcnemar_exact(up, dn)
            sig = "★" if p < 0.05 else " "
            print(f"  {SH[a]:<10}{accb:>9.1f}%{accf:>9.1f}%{accf-accb:>+8.1f}"
                  f"{up:>7}{dn:>7}{p:>9.4f}{sig}")
            details.append((SH[a], accb, accf, accf - accb, up, dn, p, common))

        print()
        deltas = [d[3] for d in details]
        sig_all = [d for d in details if d[6] < 0.05]
        print(f"  五个模型的提升幅度: {min(deltas):+.1f} 至 {max(deltas):+.1f} 百分点")
        print(f"  全部为正: {'是' if all(d > 0 for d in deltas) else '否'}")
        print(f"  统计显著: {len(sig_all)}/5 个模型")

        # 合并
        allup = sum(d[4] for d in details)
        alldn = sum(d[5] for d in details)
        pall = metrics.mcnemar_exact(allup, alldn)
        print()
        print(f"  五模型合并：变对 {allup} ／ 变错 {alldn} ／ 净 {allup-alldn:+d}")
        print(f"  McNemar p = {pall:.2e}   {'★ 极其显著' if pall < 1e-6 else '★ 显著' if pall < 0.05 else '不显著'}")

        # 分领域（合并五模型）
        dom = defaultdict(lambda: [0, 0])
        for a in ARMS:
            b = tier_map(*base, a, "lenient")
            f = tier_map(*free, a, "lenient")
            for i in sorted(set(b) & set(f)):
                inst = pool.get(i)
                if inst is None:
                    continue
                d = inst.stratum.get("subject", "?")
                if f[i] == "correct" and b[i] != "correct":
                    dom[d][0] += 1
                elif b[i] == "correct" and f[i] != "correct":
                    dom[d][1] += 1

        print()
        print(f"  {'领域':<20}{'变对':>6}{'变错':>6}{'净':>7}")
        print("  " + "-" * 42)
        for d in sorted(dom, key=lambda k: -(dom[k][0] - dom[k][1])):
            up, dn = dom[d]
            print(f"  {d:<20}{up:>6}{dn:>6}{up-dn:>+7}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
