"""跨配置配对比较：只在**共同题**上作差。

为什么不能用两次报告的数字直接相减
    C2 / C3 / C5 的题集都是 C1 的子集。拿"C2 在 150 题上的准确率"
    减"C1 在 588 题上的准确率"，是拿两个**不同总体**作差 ——
    差值里混着"这 150 题恰好更难"的效应，而不是配置因子的效应。
    配对比较把范围锁死在两侧都出现的题上，并用 McNemar 精确检验
    只看**不一致**的题，才算得清因子到底值多少。

用法
    .venv\\Scripts\\python.exe scripts\\compare_runs.py ^
        --base c1-full --base-config c1 --other c2-full --other-config c2

    # 只比一个口径
    ... --mode lenient

产物
    runs/{other}/compare_vs_{base}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import metrics, paths  # noqa: E402
from meval.types import Config, ParseMode, Tier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="跨配置配对比较")
    ap.add_argument("--base", required=True, help="基准 run_id")
    ap.add_argument("--base-config", default=None, help="基准配置键，默认由 run_id 推断")
    ap.add_argument("--other", required=True, help="对照 run_id")
    ap.add_argument("--other-config", default=None, help="对照配置键，默认由 run_id 推断")
    ap.add_argument("--mode", default=None, choices=["strict", "lenient"], help="只比一个口径")
    ap.add_argument("--json", action="store_true", help="落盘 JSON")
    args = ap.parse_args()

    def _cfg(run_id: str, explicit: str | None) -> Config:
        if explicit:
            return Config(explicit)
        head = run_id.split("-")[0]
        return Config(head)

    base_cfg = _cfg(args.base, args.base_config)
    other_cfg = _cfg(args.other, args.other_config)

    bt = metrics.load_tallies(args.base, base_cfg)
    ot = metrics.load_tallies(args.other, other_cfg)
    if not bt or not ot:
        print(f"!! tally 缺失：{args.base}/{base_cfg.value} {len(bt)} 条，"
              f"{args.other}/{other_cfg.value} {len(ot)} 条")
        return 2

    arms = sorted({t["arm"] for t in bt} & {t["arm"] for t in ot})
    modes = ([ParseMode(args.mode)] if args.mode
             else [ParseMode.STRICT, ParseMode.LENIENT])

    print("=" * 92)
    print(f"配对比较   {args.base}({base_cfg.value})  →  {args.other}({other_cfg.value})")
    print("=" * 92)

    detail: dict = {
        "base": {"run_id": args.base, "config": base_cfg.value},
        "other": {"run_id": args.other, "config": other_cfg.value},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": {},
    }

    for mode in modes:
        print(f"\n口径 = {mode.value}")
        print(f"  {'模型':<28}{'共同题':>6}{'可配对':>7}{'覆盖':>7}{'基准':>8}{'对照':>8}"
              f"{'Δpp':>8}{'基准对/对照错':>13}{'基准错/对照对':>13}{'McNemar p':>11}")
        for arm in arms:
            r = metrics.paired_compare(
                bt + ot, base_arm=arm, other_arm=arm,
                base_config=base_cfg.value, other_config=other_cfg.value, mode=mode,
            )
            detail["arms"].setdefault(arm, {})[mode.value] = r
            if not r.get("n_shared"):
                print(f"  {arm:<28}{'—':>6}   无共同题")
                continue

            cov = r["coverage"]
            if r["n_paired"] == 0:
                # 一侧完全无法二元化（典型：严格口径下 C2 全是 invalid）
                print(f"  {arm:<28}{r['n_shared']:>6}{0:>7}{cov:>7.0%}"
                      f"{'—':>8}{'—':>8}{'—':>8}{'—':>13}{'—':>13}{'不可配对':>11}")
                continue

            # p 为 None 有两种**含义完全不同**的情形，不能混着打印：
            #   n_paired == 0        -> 一侧无法二元化（上面已拦掉，不报 Δpp）
            #   n_paired > 0 且 b+c==0 -> 两侧在每一道可比题上结论都一致 = 无分歧
            # 之前这里直接 `:.2e` 格式化，遇到第二种情况会抛
            # TypeError: unsupported format string passed to NoneType.__format__
            p = r["mcnemar_p"]
            pstr = f"{p:.2e}" if p is not None else "无分歧"

            print(f"  {arm:<28}{r['n_shared']:>6}{r['n_paired']:>7}{cov:>7.0%}"
                  f"{r['acc_base']:>8.1%}{r['acc_other']:>8.1%}{r['delta_pp']:>+8.1f}"
                  f"{r['base_correct_other_not']:>13}{r['base_not_other_correct']:>13}"
                  f"{pstr:>11}")

        # 覆盖率过低必须显式警告：否则"没有差异"会被误读成"因子无效"
        lows = [(a, detail["arms"][a][mode.value]) for a in arms
                if 0 < (detail["arms"][a][mode.value].get("coverage") or 0) < 0.5]
        if lows:
            print("\n  !! 覆盖率 < 50%：" + "；".join(
                f"{a} {r['coverage']:.0%}（因 {r['n_offscale']} 条无二元判定" +
                (f"、{r['n_tie_reps']} 条重复采样并列" if r["n_tie_reps"] else "") + "）"
                for a, r in lows))
            print("     覆盖率低时 McNemar 只看少数不一致题，结论要谨慎表述。")

    # 汇总：把"某侧完全不可配对"这件事说清楚
    print("\n" + "=" * 92)
    print("读法")
    print("=" * 92)
    print("  Δpp = 对照 − 基准（同题配对，非两次报告数字相减）")
    print("  McNemar 只用两侧都能二元化、且结论不一致的题；两边都对/都错的题不含信息。")
    print("  覆盖率为 0 表示该口径下一侧全部无法判出对错（典型：严格口径遇上自由作答）。")

    if args.json:
        out = paths.ROOT / "runs" / args.other / f"compare_vs_{args.base}.json"
        out.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
