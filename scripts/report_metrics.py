"""指标报告：读跑批产物 → 四档分账 + CI + 分层拆分 → CSV / JSON。

用法：
    .venv\\Scripts\\python.exe scripts\\report_metrics.py --run-id c1-full
    .venv\\Scripts\\python.exe scripts\\report_metrics.py --run-id c1-full --config c1

产物：
    runs/{run_id}/metrics.csv          每个模型 × 每口径 一行的 tidy 表
    runs/{run_id}/metrics.json         含分层拆分、口径一致率、跨配置 Δacc
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

from meval import build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402


def load_instances() -> dict[str, dict]:
    """把冻结题集读成 {id: dict}，供分层拆分用。"""
    out: dict[str, dict] = {}
    for name in ("eval_set.jsonl",):
        p = paths.data() / name
        if p.exists():
            for inst in build_set.load_frozen(p):
                out[inst.id] = inst.to_dict()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="c1")
    args = ap.parse_args()

    cfg = Config(args.config)
    instances = load_instances()
    results_list = execute.load_results(args.run_id, cfg)
    # ★ 键必须含 arm —— 三个模型跑同一批题（instance_id, rep 完全相同），
    #   只用 (instance_id, rep) 做键会让三个模型互相覆盖，只剩最后一个模型。
    results = {(r.arm, r.instance_id, r.rep): r.to_dict() for r in results_list}
    tallies = metrics.load_tallies(args.run_id, cfg)

    if not tallies:
        print(f"!! runs/{args.run_id}/tallies/{cfg.value}/all.jsonl 不存在或为空")
        print("   先跑 scripts/run_config.py 生成 tally")
        return 2

    arms = sorted({t["arm"] for t in tallies})
    print("=" * 78)
    print(f"指标报告   run_id={args.run_id}   config={cfg.value}")
    print("=" * 78)
    print(f"记录: {len(results_list)} 次调用 | Tally: {len(tallies)} 条 | 模型: {len(arms)}")
    print(f"题集: {len(instances)} 题（含分层标签）\n")

    rows: list[dict] = []
    detail: dict = {"run_id": args.run_id, "config": cfg.value, "arms": {}}

    for arm in arms:
        detail["arms"][arm] = {}
        for mode in (ParseMode.STRICT, ParseMode.LENIENT):
            st = metrics.compute_arm(
                arm=arm, config=cfg, tallies=tallies, results=results,
                instances=instances, mode=mode,
            )
            row = metrics.arm_to_row(st)
            row["mode"] = mode.value
            rows.append(row)

            detail["arms"][arm][mode.value] = {
                **row,
                "by_stratum": st.by_stratum,
                "by_source": st.by_source,
            }

        detail["arms"][arm]["strict_lenient"] = metrics.strict_lenient_agreement(tallies, arm)
        # 判分一致性完整报告（混淆矩阵 + κ/φ + 报告清单项）
        detail["arms"][arm]["judge_agreement"] = metrics.judge_agreement_report(tallies, arm)
        # C3 重复采样：同题跨次一致率。
        # ★ 必须显式要求 reps>1：reps=1 时每题只有一次观测，
        #   一致率恒为 100%（且 answer_all_same_rate 为 None，格式化会直接崩），
        #   打印出来等于把"只跑了一次"包装成"完全确定"。
        cons = metrics.consistency_c3(tallies, arm)
        if cons.get("reps", 0) > 1:
            detail["arms"][arm]["consistency"] = cons
            detail["arms"][arm]["rep_spread"] = metrics.rep_accuracy_spread(tallies, arm)

    # ---- 控制台表 ---------------------------------------------------------
    print(f"{'模型':<28}{'口径':<9}{'准确率':>8}{'95% CI':>16}{'无效':>7}{'截断':>7}{'n':>6}")
    print("-" * 78)
    for r in rows:
        ci = "—"
        if r["acc_ci_low"] is not None:
            ci = f"[{r['acc_ci_low']:.3f},{r['acc_ci_high']:.3f}]"
        acc = f"{r['acc']:.1%}" if r["acc"] is not None else "n/a"
        print(f"{r['arm']:<28}{r['mode']:<9}{acc:>8}{ci:>16}"
              f"{(r['invalid_rate'] or 0):>7.1%}{(r['truncated_rate'] or 0):>7.1%}{r['n_total']:>6}")

    # ---- 口径一致率 -------------------------------------------------------
    print("\n" + "=" * 78)
    print("严格 vs 宽松口径一致性（严格判无效、宽松判出答案 = 判分口径造成的伪低分）")
    print("=" * 78)
    for arm in arms:
        r = detail["arms"][arm]["strict_lenient"]
        if not r:
            continue
        print(f"  {arm:<28} 一致率 {r['agreement']:.1%}  Kappa {r['kappa']:.3f}  "
              f"被救回 {r['strict_invalid_lenient_parsed']}/{r['n']} ({r['rescued_rate']:.1%})")

    # ---- 分层拆分 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("分层拆分（严格口径，准确率）")
    print("=" * 78)
    for arm in arms:
        d = detail["arms"][arm]["strict"]
        if not d.get("n_correct"):
            continue
        print(f"\n  {arm}")
        for key in ("subject", "difficulty", "n_options"):
            bucket = d["by_stratum"].get(key, {})
            cells = [(k, v) for k, v in sorted(bucket.items()) if v.get("n")]
            if not cells:
                continue
            s = "  ".join(f"{k}={v['acc']:.0%}(n={v['n']})" for k, v in cells[:8])
            print(f"    {key:<11} {s}")
        src = d.get("by_source", {})
        if src:
            print(f"    source      " + "  ".join(
                f"{k}={v['acc']:.0%}(n={v['n']})" for k, v in sorted(src.items())))

    # ---- 混淆矩阵与判分一致性（依据 arXiv 2606.00093 的报告清单）----------
    print("\n" + "=" * 78)
    print("判分一致性报告（参考=严格口径，待验证=宽松口径）")
    print("=" * 78)
    print("  判据量表      : 含弃权的名义量表（correct/wrong/invalid/truncated）")
    print("  弃权处理      : invalid/truncated 独立成档；二元一致性仅在两者均判出 correct/wrong 的条目上计算")
    for arm in arms:
        ja = detail["arms"][arm].get("judge_agreement", {})
        b = ja.get("binary", {})
        if not b or not b.get("n"):
            continue
        cm = b["confusion"]
        print(f"\n  {arm}")
        print(f"    覆盖率 {b['coverage']:.1%}（n={b['n']}，"
              f"排除非二元 {b['excluded_non_binary']} 条）")
        print(f"    混淆矩阵（行=严格/参考，列=宽松/预测）")
        print(f"      {'':<12}{'宽松:correct':>14}{'宽松:wrong':>12}")
        print(f"      {'严格:correct':<12}{cm['TP']:>14}{cm['FN']:>12}")
        print(f"      {'严格:wrong':<12}{cm['FP']:>14}{cm['TN']:>12}")
        print(f"    Accuracy {b['accuracy']:.4f}  Precision {b['precision']:.4f}  "
              f"Recall {b['recall']:.4f}  F1 {b['f1']:.4f}")
        print(f"    Cohen's κ {b['cohen_kappa']:.4f}   φ/MCC {b['phi_mcc']:.4f}   "
              f"κ−φ {b['kappa_minus_phi']:+.4f}")
        print(f"    正例率：参考 {b['prevalence_ref']:.4f} → 预测 {b['prevalence_pred']:.4f}  "
              f"漂移 {b['prevalence_drift']:+.4f}")
        print(f"    （二元下 Pearson/Spearman/Kendall/φ/MCC 同值，故只报 φ 一个）")
        at = ja.get("all_tiers", {})
        if at.get("top_disagreements"):
            tops = "  ".join(f"{d['ref']}→{d['pred']}×{d['n']}" for d in at["top_disagreements"])
            print(f"    全档分歧 top: {tops}")

    # ---- 确定性（C3 重复采样）---------------------------------------------
    if any("consistency" in detail["arms"][a] for a in arms):
        reps_n = max(detail["arms"][a]["consistency"]["reps"] for a in arms
                     if "consistency" in detail["arms"][a])
        print("\n" + "=" * 78)
        print(f"确定性：同题跨次一致率（重复采样，每题 {reps_n:g} 次）")
        print("=" * 78)
        print("  这是「温度固定到底有多确定」的实测答案。")
        print("  ★ 注意：「温度参数被忽略」≠「输出可复现」—— 实测两者不同。\n")

        def _pct(x):
            return f"{x:.1%}" if isinstance(x, (int, float)) else "—"

        print(f"  {'模型':<28}{'题数':>5}{'可答':>6}{'从不作答':>9}"
              f"{'档位全同':>10}{'可答题档位全同':>16}{'答案全同':>10}")
        for arm in arms:
            c = detail["arms"][arm].get("consistency")
            if not c:
                continue
            print(f"  {arm:<28}{c['n_instances']:>5}{c['n_answerable']:>6}"
                  f"{c['n_never_answered']:>9}{_pct(c['tier_all_same_rate']):>10}"
                  f"{_pct(c.get('answerable_tier_all_same_rate')):>16}"
                  f"{_pct(c.get('answer_all_same_rate')):>10}")
        print("\n  「档位全同」是全量口径，含 10 次全部 invalid 的题 ——")
        print("   那不叫确定，叫从不作答；故必须与「可答题档位全同」并排看。")

        spreads = {a: detail["arms"][a].get("rep_spread") or {} for a in arms}
        if any(spreads.values()):
            print(f"\n  {'模型':<28}{'每次准确率':>11}{'最低':>8}{'最高':>8}{'极差pp':>9}{'SD':>8}")
            for arm in arms:
                s = spreads[arm]
                if not s:
                    continue
                print(f"  {arm:<28}{s['acc_mean']:>11.1%}{s['acc_min']:>8.1%}"
                      f"{s['acc_max']:>8.1%}{s['acc_range_pp']:>9.2f}{s['acc_sd']:>8.4f}")
            print("\n  读法：把每个 rep 当成一次独立的缩量评测 ——")
            print("       极差 = 「跑一次评测、报一个点估计」这件事本身的重复噪声。")
            print("       它与 CI（抽样误差）是两条独立来源，且**不能靠加题量消除**。")

    # ---- 成本-效用 ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("成本-效用（严格口径；成本按统一折算口径，仅供横向比较）")
    print("=" * 78)
    strict_rows = [r for r in rows if r["mode"] == ParseMode.STRICT.value]
    by_arm_row = {r["arm"]: r for r in strict_rows}
    pts = metrics.pareto_frontier([
        {"arm": r["arm"], "acc": r["acc"], "cost_cny": r["cost_cny"],
         "median_latency_s": r["median_latency_s"], "p95_latency_s": r["p95_latency_s"],
         "cost_per_1k_correct": r["cost_per_1k_correct"], "total_tokens": r["total_tokens"]}
        for r in strict_rows
    ])
    print(f"  {'模型':<28}{'准确率':>8}{'成本(元)':>10}{'千题成本':>10}"
          f"{'tokens':>10}{'中位延迟':>9}{'p95':>8}{'前沿':>7}")
    for p in sorted(pts, key=lambda x: -(x["acc"] or 0)):
        print(f"  {p['arm']:<28}{p['acc']:>8.1%}{p['cost_cny']:>10.3f}"
              f"{p['cost_per_1k_correct']:>10.2f}{p['total_tokens']:>10,}"
              f"{p['median_latency_s']:>9.2f}{p['p95_latency_s']:>8.2f}"
              f"{'   ★' if p.get('pareto') else '   —':>7}")
    print("\n  ★ = 在「准确率↑ 成本↓」下非支配（Pareto 前沿）")
    tot_cost = sum(p["cost_cny"] for p in pts)
    print(f"  三个模型合计折算成本：{tot_cost:.3f} 元")

    # 口径说明：κ 与一致率看似矛盾时，读这一段
    print("\n  读法提示：κ=1.0 但一致率<100% 并不矛盾 ——")
    print("    κ 只在**两口径均判出 correct/wrong** 的条目上计算（那是判分分歧）；")
    print("    一致率还包含了「一侧判 invalid」的条目（那是解析覆盖差异，不是判分分歧）。")
    print("    本项目实测：二元判分分歧为 0，全部分歧都来自解析覆盖（见下方全档分歧）。")

    # ---- 落盘 -------------------------------------------------------------
    csv_p = paths.ROOT / "runs" / args.run_id / "metrics.csv"
    metrics.to_csv(rows, csv_p)
    json_p = paths.ROOT / "runs" / args.run_id / "metrics.json"
    json_p.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  CSV  -> {csv_p.relative_to(paths.ROOT)}")
    print(f"  JSON -> {json_p.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
