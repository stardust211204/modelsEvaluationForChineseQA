"""C3 稳定性分析的 CLI 封装。

★ 这里**不重新实现口径** —— 一律调用 `meval.metrics.consistency_c3`。
  教训：本脚本的第一版自己写了一遍"答案全同率"，结果与库里已有的实现
  口径不一致，还让我误以为发现了缺陷。**同一个指标绝不能有两个实现。**

`consistency_c3` 给三个口径，用途不同：

    tier_all_same_rate              全量档位全同 —— **不要用它下结论**
                                    一道 10 次全 invalid 的题档位"当然全同"，
                                    于是「永远不吐答案」被算成「非常确定」
    answerable_tier_all_same_rate   只统计至少答出一次的题
    ★ answer_all_same_rate          答案层面是否每次都一样（结论口径）
                                    invalid 的次数被排除在外

用法
    .venv\\Scripts\\python.exe scripts\\analyze_stability.py
    .venv\\Scripts\\python.exe scripts\\analyze_stability.py --c3-run c3-5arm-v2 --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import exclusions, metrics, paths  # noqa: E402
from meval.types import Config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="C3 稳定性分析（调用库函数，保证单一口径）")
    ap.add_argument("--c3-run", default="c3-5arm-v2")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    run = args.c3_run
    tpath = paths.tallies_dir(run, Config.C3.value) / "all.jsonl"
    if not tpath.exists():
        print(f"!! {run} 没有 C3 tallies：{tpath}")
        return 2

    tallies = metrics.load_tallies(run, Config.C3)
    if not tallies:
        print(f"!! {run} 的 C3 tallies 为空")
        return 2

    drop = exclusions.load()
    print("=" * 96)
    print(f"C3 稳定性  run={run}（严格口径，调用 metrics.consistency_c3）")
    print("=" * 96)
    print(f"  {exclusions.summary()}")
    print()
    print(f"  {'模型':<28}{'n答过':>7}{'从不作答':>9}{'★答案全同':>11}"
          f"{'答过档位全同':>14}{'全量档位全同':>14}")
    print("  " + "-" * 78)

    rows = []
    for arm in sorted({t["arm"] for t in tallies}):
        sub = [t for t in tallies if t.get("instance_id") not in drop]
        r = metrics.consistency_c3(sub, arm)
        if not r:
            continue
        print(f"  {arm[:26]:<28}{r['n_answerable']:>7}{r['n_never_answered']:>9}"
              f"{(r['answer_all_same_rate'] or 0):>11.1%}"
              f"{(r['answerable_tier_all_same_rate'] or 0):>14.1%}"
              f"{r['tier_all_same_rate']:>14.1%}")
        rows.append({"arm": arm, **r})

    print()
    print("  读法")
    print("  * **★答案全同 = 结论口径**：只问「解出来的选项每次是不是同一个」，")
    print("    把解析不出的次数排除在外（那属于格式问题，不是答案不稳）。")
    print("  * `从不作答` 本身是一条竞品信息：它度量「格式可控性」——")
    print("    有些模型有若干题**每次都答不出可解析的答案**。")
    print("  * `全量档位全同` 仅用于与文献口径对照，**不可作为结论**。")

    if args.write:
        out = paths.ROOT / "runs" / run / "stability.json"
        out.write_text(json.dumps({"run": run, "arms": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
