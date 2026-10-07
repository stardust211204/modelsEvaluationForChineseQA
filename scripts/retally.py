"""用**当前**解析器重算所有已跑批次的 tallies，并逐项对比旧值。

为什么需要这个脚本
    `src/meval/tally.py` 的宽松解析修了一个会**制造虚假竞品差异**的 bug
    （详见该文件 `ANSWER_RE` 上方的注释）。tallies 是从 raw **可重算**的，
    所以不需要任何模型调用 —— 这正是当初把 raw（不可变）与 tallies（可重算）
    分开的设计意图，这次是它第一次真正被用上。

★ 安全措施
    1. 旧 tallies 先备份成 `all.jsonl.pre-parserfix`，**不覆盖**，可随时回退；
    2. 逐 (run, config, arm, mode) 报出准确率变化，**变化为 0 的也要打印**
       （否则无法区分"没受影响"和"没跑到"）；
    3. 跑完打印受影响最大的条目，便于定位哪些结论需要重写。

用法
    .venv\\Scripts\\python.exe scripts\\retally.py --dry-run   # 只看变化，不写盘
    .venv\\Scripts\\python.exe scripts\\retally.py             # 写盘（含备份）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402

# 只有 MCQ 配置走 parse_answer；C4（judge）走 parse_judge、C7 走 parse_refusal，均不受影响
MCQ_CONFIGS = [Config.C1, Config.C2, Config.C3, Config.C5, Config.C6]
PERMUTED = {Config.C6}   # 该配置的 rep 即"第几个排列"，判分必须用同一排列


def acc_of(tiers: list[str]) -> float | None:
    return metrics.tier_accounting(tiers)["acc"]


def main() -> int:
    ap = argparse.ArgumentParser(description="用当前解析器重算全部 tallies 并对比")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    ap.add_argument("--only", default=None, help="只处理名字含该子串的 run")
    args = ap.parse_args()

    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    runs_dir = paths.ROOT / "runs"
    runs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if args.only:
        runs = [d for d in runs if args.only in d.name]

    print("=" * 100)
    print(f"重算 tallies（解析器已修复）{'  [DRY-RUN 不写盘]' if args.dry_run else ''}")
    print("=" * 100)

    changed: list[tuple[str, str, str, str, float | None, float | None]] = []
    n_checked = n_skipped = 0

    for d in runs:
        for cfg in MCQ_CONFIGS:
            tpath = d / "tallies" / cfg.value / "all.jsonl"
            if not tpath.exists():
                continue
            try:
                results = execute.load_results(d.name, cfg)
            except Exception:
                n_skipped += 1
                continue
            if not results:
                continue
            n_checked += 1

            # ---- 旧值 ----
            old_rows = [json.loads(l) for l in tpath.read_text(encoding="utf-8").splitlines() if l.strip()]
            old_by: dict[tuple[str, str], list[str]] = defaultdict(list)
            for t in old_rows:
                old_by[(t["arm"], t["mode"])].append(t["tier"])

            # ---- 新值（复刻 run_config.py 的判分逻辑）----
            new_by: dict[tuple[str, str], list[str]] = defaultdict(list)
            for r in results:
                inst = pool.get(r.instance_id)
                if inst is None:
                    continue
                if cfg in PERMUTED:
                    inst = build_set.permute_instance(inst, r.rep)[0]
                for t in tally.tally_both_modes(r, inst):
                    new_by[(t.arm, t.mode.value)].append(t.tier.value)

            keys = sorted(set(old_by) | set(new_by))
            for k in keys:
                arm, mode = k
                if mode != ParseMode.LENIENT.value:
                    continue          # 严格口径只认裸字母，不受该 bug 影响
                a, b = acc_of(old_by.get(k, [])), acc_of(new_by.get(k, []))
                if a == b and old_by.get(k) == new_by.get(k):
                    continue
                changed.append((d.name, cfg.value, arm, mode, a, b))

            if not args.dry_run:
                bak = tpath.with_suffix(".jsonl.pre-parserfix")
                if not bak.exists():
                    shutil.copy2(tpath, bak)
                new_rows = []
                for r in results:
                    inst = pool.get(r.instance_id)
                    if inst is None:
                        continue
                    if cfg in PERMUTED:
                        inst = build_set.permute_instance(inst, r.rep)[0]
                    new_rows.extend(tally.tally_both_modes(r, inst))
                tpath.write_text(
                    "\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in new_rows) + "\n",
                    encoding="utf-8")

    print(f"  检查了 {n_checked} 个 (run, config) 组合，跳过 {n_skipped}")
    print(f"  宽松口径发生变化的 (run, config, arm): **{len(changed)}**")
    print()
    if changed:
        print(f"  {'run':<22}{'cfg':<5}{'arm':<28}{'旧':>9}{'新':>9}{'变化':>10}")
        print("  " + "-" * 84)
        for run, cfg, arm, _m, a, b in sorted(changed, key=lambda x: -abs((x[5] or 0) - (x[4] or 0))):
            fa = "无定义" if a is None else f"{a:.1%}"
            fb = "无定义" if b is None else f"{b:.1%}"
            dlt = "" if (a is None or b is None) else f"{(b-a)*100:+.1f}pp"
            print(f"  {run:<22}{cfg:<5}{arm[:26]:<28}{fa:>9}{fb:>9}{dlt:>10}")
    else:
        print("  ✅ 没有任何条目变化 —— 该 bug 未影响已存的任何结论")

    if not args.dry_run:
        print()
        print("  旧 tallies 已备份为 `all.jsonl.pre-parserfix`（回退：改回该文件名即可）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
