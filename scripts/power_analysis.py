"""题量该多大：基于**实测数据**的功效分析。

为什么不能用 ±4pp 的边际 CI 来定题量
    模型排名靠的是**配对**比较（同题对同题），而配对检验只用到**分歧对**。
    拿边际 CI 半宽 4pp 去推，会得出"5pp 的差距测不出来"——
    实测并非如此：588 题时配对 McNemar 在 80% 功效下已能检出约 5pp。

公式（McNemar 精确检验，双侧 α=0.05，功效 1−β=0.80）
    设 δ = 真实准确率差，π_d = 分歧率（两个模型作答不同的题占比）。
    分歧格 b、c 满足 b+c ≈ n·π_d，b−c ≈ n·δ，
    统计量 z ≈ δ·√(n/π_d)，要求 z ≥ z_{α/2} + z_β = 1.96 + 0.84 = 2.80。

        给定 n 时的最小可检出差距：  δ_min = 2.80 · √(π_d / n)
        给定 δ 时所需题量：          n_req = π_d · (2.80 / δ)²

★ π_d 全部取自**实测**（C1 的 588 题），不是拍的。

用法
    .venv\\Scripts\\python.exe scripts\\power_analysis.py
    .venv\\Scripts\\python.exe scripts\\power_analysis.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, metrics, paths  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402

Z = 1.96 + 0.84          # α=0.05 双侧 + 80% 功效
STRICT = ParseMode.STRICT.value


def min_detectable(pi_d: float, n: int) -> float:
    """给定题量与分歧率，配对检验在 80% 功效下能检出的最小准确率差。"""
    if n <= 0 or pi_d <= 0:
        return float("nan")
    return Z * (pi_d / n) ** 0.5


def n_required(pi_d: float, delta: float) -> float:
    """要检出 delta 的差距，需要多少题。"""
    if delta <= 0:
        return float("inf")
    return pi_d * (Z / delta) ** 2


def main() -> int:
    ap = argparse.ArgumentParser(description="题量功效分析")
    ap.add_argument("--run-id", default="c1-full")
    ap.add_argument("--config", default="c1")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = Config(args.config)
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    rows = [t for t in metrics.load_tallies(args.run_id, cfg) if t["mode"] == STRICT]
    if not rows:
        print(f"!! runs/{args.run_id} 无 {STRICT} 口径 tally")
        return 2

    # (arm, instance) -> tier；只用可判条目
    ans: dict[str, dict[str, str]] = defaultdict(dict)
    for t in rows:
        if t["tier"] in ("correct", "wrong"):
            ans[t["arm"]][t["instance_id"]] = t["tier"]
    arms = sorted(ans)
    common = set.intersection(*(set(v) for v in ans.values())) if ans else set()
    n_common = len(common)

    print("=" * 80)
    print(f"题量功效分析   run={args.run_id}   同题可判 {n_common} 条   模型 {len(arms)}")
    print("=" * 80)
    print(f"  公式：δ_min = {Z:.2f}·√(π_d/n)   ｜  n_req = π_d·({Z:.2f}/δ)²")
    print("  π_d（分歧率）全部取自实测，不是假设值。\n")

    # ---- 整体：各模型对的分歧率与当前可检出差距 ------------------------------
    print(f"  {'模型对':<46}{'π_d':>8}{'实测差距':>10}{'δ_min':>9}{'可检出?':>9}")
    out: dict = {"run_id": args.run_id, "n_common": n_common, "pairs": {}}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            disc = sum(1 for k in common if ans[a][k] != ans[b][k])
            pi_d = disc / n_common
            acc_a = sum(1 for k in common if ans[a][k] == "correct") / n_common
            acc_b = sum(1 for k in common if ans[b][k] == "correct") / n_common
            delta = abs(acc_a - acc_b)
            dmin = min_detectable(pi_d, n_common)
            ok = "✅ 能" if delta >= dmin else "❌ 不能"
            print(f"  {a + ' vs ' + b:<46}{pi_d:>8.1%}{delta:>10.1%}"
                  f"{dmin:>9.1%}{ok:>9}")
            out["pairs"][f"{a}|{b}"] = {
                "pi_d": round(pi_d, 4), "observed_delta": round(delta, 4),
                "delta_min_detectable": round(dmin, 4),
                "detectable": bool(delta >= dmin),
            }

    # ---- 分层：真正被卡住的地方 -------------------------------------------
    by_subj: dict[str, list[str]] = defaultdict(list)
    for k in common:
        by_subj[(pool[k].stratum or {}).get("subject", "unknown")].append(k)

    # 取分歧最多的一对作为代表性比较
    pair = max(out["pairs"], key=lambda p: out["pairs"][p]["pi_d"]).split("|")
    print()
    print(f"  分层（以分歧最多的一对 {pair[0]} vs {pair[1]} 为例）：")
    print(f"  {'学科':<16}{'n':>5}{'π_d':>8}{'实测差距':>10}{'δ_min':>9}"
          f"{'→10pp 需要 n':>15}")
    subs: dict = {}
    for subj, ks in sorted(by_subj.items(), key=lambda kv: -len(kv[1])):
        n = len(ks)
        disc = sum(1 for k in ks if ans[pair[0]][k] != ans[pair[1]][k])
        pi_d = disc / n if n else 0.0
        acc0 = sum(1 for k in ks if ans[pair[0]][k] == "correct") / n if n else 0
        acc1 = sum(1 for k in ks if ans[pair[1]][k] == "correct") / n if n else 0
        delta = abs(acc0 - acc1)
        dmin = min_detectable(pi_d, n) if pi_d else float("nan")
        need = n_required(pi_d, 0.10) if pi_d else float("nan")
        print(f"  {subj:<16}{n:>5}{pi_d:>8.1%}{delta:>10.1%}{dmin:>9.1%}"
              f"{need:>15.0f}")
        subs[subj] = {"n": n, "pi_d": round(pi_d, 4),
                      "observed_delta": round(delta, 4),
                      "delta_min_detectable": round(dmin, 4) if dmin == dmin else None,
                      "n_needed_for_10pp": round(need) if need == need else None}
    out["by_subject"] = subs
    out["representative_pair"] = pair

    # ---- 建议 --------------------------------------------------------------
    mean_pi = sum(v["pi_d"] for v in out["pairs"].values()) / max(1, len(out["pairs"]))
    n_subj = max(1, len(by_subj))
    print()
    print("=" * 80)
    print("建议")
    print("=" * 80)
    print(f"  实测平均分歧率 π_d = {mean_pi:.1%}；学科数 = {n_subj}")
    print()
    print(f"  {'目标可检出差距':<18}{'所需总题量':>12}{'相对现状':>10}{'调用量(×3 个模型)':>14}")
    for target in (0.05, 0.04, 0.03, 0.025, 0.02):
        n_req = n_required(mean_pi, target)
        print(f"  {f'{target:.1%}':<18}{n_req:>12.0f}{n_req/n_common:>9.1f}×"
              f"{n_req*3:>14.0f}")
    print()
    print("  分层（每学科各自达到 10pp 可检出）：")
    need_per_subj = n_required(mean_pi, 0.10)
    print(f"    每学科需 {need_per_subj:.0f} 题 × {n_subj} 学科 ≈ "
          f"{need_per_subj*n_subj:.0f} 题（{need_per_subj*n_subj/n_common:.1f}× 现状）")
    out["recommendation"] = {
        "mean_pi_d": round(mean_pi, 4),
        "n_subjects": n_subj,
        "n_for_10pp_per_subject": round(need_per_subj * n_subj),
        "n_current": n_common,
    }

    print()
    print("  读法：")
    print("   * 头条结论在 588 题上**已经落定**，扩规模不改变它们。")
    print("   * 被卡住的是**分层**：每学科 n 只有 20–70，可检出差距在 20pp 量级。")
    print("   * 「10 倍」这个数字来自边际 CI；按**配对**口径算，所需题量小得多。")

    if args.write:
        p = paths.ROOT / "runs" / args.run_id / "power_analysis.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON -> {p.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
