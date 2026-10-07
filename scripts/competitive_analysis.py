"""竞品分析聚合：把各臂的**能力 / 稳定性 / 鲁棒性 / 成本效率**拼成一张可比的表。

为什么单独写一个脚本
    现有 `report_metrics.py` 是按**配置**组织的（C1 一节、C2 一节），输出的是
    "方法学"视角。竞品分析需要的是**按臂**组织的视角：每个模型一行，维度是列，
    能直接横向比 —— 那就是"模型画像卡"。

口径纪律（每条都对应一次踩过的坑）
    * 准确率走 `metrics.tier_accounting`（项目唯一四档口径），严格/宽松都出
    * **题集排除清单生效**：`meval.exclusions` 里那 37 道 LaTeX 提取错误的题
      从**所有**统计指标里剔除（题集文件不动、哈希不变、可回溯）
    * **稳定性用修正口径**：只统计所有 rep 都解出答案的题。
      原口径会把"每次都解析不出"算成"答案一致" ⇒ 越差越显得稳
    * 成本 = 统一折算价（入 1 元/M、出 4 元/M）。**这不是账单** ——
      聚合端点计费不可见；真实价格跑完后单独查表
    * 截断率本身是一个竞品维度：同样 max_tokens 下谁更早撞上限 = 谁更烧预算
    * **跨 run 的臂不顶替**：取不到就显式标 `—`，绝不用另一个 run 的同名臂凑数

用法
    .venv\\Scripts\\python.exe scripts\\competitive_analysis.py
    .venv\\Scripts\\python.exe scripts\\competitive_analysis.py --json out.json
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

from meval import build_set, exclusions, metrics, paths, tally  # noqa: E402
from meval.types import ParseMode  # noqa: E402

PRICE_IN, PRICE_OUT = 1.0, 4.0   # 元 / 百万 token（统一折算价，非账单）

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SHORT = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
         "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
         "qwen3.8-flash": "qwen"}

# 显示名: (run_id, tallies 目录名)
CONFIGS = {
    "C1 / MMLU-ProX":    ("c1-5arm-v2", "c1"),
    "C1 / C-Eval":       ("ceval-c1-5arm-v2", "c1"),
    "C2-hi / MMLU-ProX": ("c2hi-5arm-v2", "c2"),
    "C2-hi / C-Eval":    ("ceval-c2hi-5arm-v2", "c2"),
    "C3 / MMLU-ProX":    ("c3-5arm-v2", "c3"),
    "C6 / MMLU-ProX":    ("c6-5arm-v2", "c6"),
    "C5 / MMLU-ProX":    ("c5-5arm-v2", "c5"),
}


def _read_tallies(run: str, cfg: str) -> list[dict]:
    f = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_raw(run: str, cfg: str, arm: str) -> list[dict]:
    f = paths.ROOT / "runs" / run / "raw" / cfg / f"{arm}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def collect(run: str, tkey: str) -> dict[str, dict]:
    """返回 {arm: 各维度}。缺失的臂**不伪造**，直接不出现。"""
    drop = exclusions.load()
    out: dict[str, dict] = {}
    tallies = _read_tallies(run, tkey)
    for arm in ARMS:
        rows = [t for t in tallies if t.get("arm") == arm and t.get("instance_id") not in drop]
        if not rows:
            continue
        d: dict = {"run": run, "n_rows": len(rows)}
        for mode in ("strict", "lenient"):
            sub = [t["tier"] for t in rows if t.get("mode") == mode]
            if not sub:
                continue
            a = metrics.tier_accounting(sub)
            d[f"acc_{mode}"] = a["acc"]
            d[f"gradeable_{mode}"] = a["n_gradeable"]
            d[f"invalid_{mode}"] = a["invalid_rate"]
            d[f"truncated_{mode}"] = a["truncated_rate"]
            d["n_total"] = len(sub)

        raw = [r for r in _read_raw(run, tkey, arm)
               if r.get("success") and r.get("instance_id") not in drop]
        if raw:
            lat = sorted(r.get("latency_s") or 0 for r in raw)
            ti = sum(r.get("tokens_in") or 0 for r in raw)
            to = sum(r.get("tokens_out") or 0 for r in raw)
            outs = sorted(r.get("tokens_out") or 0 for r in raw)
            d.update({
                "n_raw": len(raw),
                "tokens_in": ti, "tokens_out": to,
                "median_tokens_out": outs[len(outs) // 2],
                "median_latency_s": lat[len(lat) // 2],
                "p95_latency_s": lat[min(int(len(lat) * 0.95), len(lat) - 1)],
                "cost_cny": ti / 1e6 * PRICE_IN + to / 1e6 * PRICE_OUT,
                "hard_truncated_rate": sum(1 for r in raw
                                           if r.get("finish_reason") == "length") / len(raw),
            })
        out[arm] = d
    return out


def stability(run: str, tkey: str = "c3", n_reps: int = 10) -> dict[str, dict]:
    """修正口径的稳定性：只统计所有 rep 都解出答案的题。"""
    data: dict[str, dict[str, dict[int, int | None]]] = defaultdict(lambda: defaultdict(dict))
    for t in _read_tallies(run, tkey):
        if t.get("mode") == "strict":
            data[t["arm"]][t["instance_id"]][t["rep"]] = t["answer_index"]
    drop = exclusions.load()
    out = {}
    for arm, items in data.items():
        naive = fixed = usable = tot = 0
        for iid, reps in items.items():
            if iid in drop or len(reps) != n_reps:
                continue
            tot += 1
            vals = list(reps.values())
            if len(set(vals)) == 1:
                naive += 1
            clean = [v for v in vals if v is not None]
            if len(clean) == n_reps:
                usable += 1
                if len(set(clean)) == 1:
                    fixed += 1
        if tot:
            out[arm] = {"naive": naive / tot,
                        "fixed": fixed / usable if usable else float("nan"),
                        "usable": usable, "n_items": tot}
    return out


def spacebunny_section() -> dict | None:
    """★ space-bunny 单独成节 —— **不进主表**。

    为什么不进主表
        它的思考是**强制开启的**（实测 5 种关思考写法全部无效，`reasoning_effort`
        与 `thinking` 直接 400）。而主表五臂的思考**都被我们关掉了**。
        把它放进同一张表意味着「臂」和「思考模式」同时变化 ——
        违反项目「每个配置只改一个因子」的纪律，且成本维度完全失真
        （它每次多烧 1,000+ 思考 token，其余臂只烧 1 个）。

    怎么呈现
        给**区间**而不是点估计。因为剔除截断项会**高估**准确率：
        实测被截断的题系统性更难（同批五臂 35.4% vs 未截断 60.4%，差 25pp）。
            acc_lower  只算可判项（现行四档口径，偏乐观）
            acc_upper  把截断项按难度外推（偏保守）
        `truncated_rate` 必须同时报 —— 它本身是一条竞品结论：
        **一个模型在简单格式约束下 13% 的请求拿不到输出，这是生产可用性问题。**
    """
    cands = [
        ("MMLU-ProX (C1)", "sb-c1-eval", "c1", "c1-5arm-v2", "c1"),
        ("C-Eval (C1)", "sb-c1-ceval", "c1", "ceval-c1-5arm-v2", "c1"),
    ]
    out: dict = {}
    for label, sb_run, sb_cfg, ref_run, ref_cfg in cands:
        f = paths.ROOT / "runs" / sb_run / "raw" / sb_cfg / "space-bunny-free.jsonl"
        if not f.exists():
            continue
        pool_path = (paths.data() / ("ceval_set.jsonl" if "ceval" in sb_run else "eval_set.jsonl"))
        pool = {i.id: i for i in build_set.load_frozen(pool_path)}
        drop = exclusions.load()
        rows = []
        for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()):
            if not r.get("success"):
                continue
            inst = pool.get(r.get("instance_id"))
            if inst is None or inst.id in drop:
                continue
            idx, _ = tally.parse_answer(r.get("content") or "", len(inst.options),
                                        ParseMode.LENIENT)
            tier = ("correct" if (idx is not None and idx in inst.gold_indices)
                    else ("wrong" if idx is not None else "truncated"))
            rows.append({"arm": "space-bunny-free", "mode": "strict",
                         "tier": tier, "instance_id": inst.id})
        if not rows:
            continue
        ref = collect(ref_run, ref_cfg)
        diff = metrics.item_difficulty(_read_tallies(ref_run, ref_cfg))
        b = metrics.truncation_bias(rows, "space-bunny-free", diff)
        # 主表五臂在同一题集上的最好成绩，作参照（不是同一配置，只作标尺）
        best = max((v["acc_lenient"] for v in ref.values() if v.get("acc_lenient") is not None),
                   default=None)
        out[label] = {**b, "ref_best_main": best, "n_answered": len(rows),
                      "ref_run": ref_run}
    return out or None


def fmt(v, pct=False, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "n/a"
    return f"{v:.1%}" if pct else (f"{v:,.{nd}f}" if isinstance(v, float) else f"{v:,}")


def main() -> int:
    ap = argparse.ArgumentParser(description="五臂竞品分析聚合")
    ap.add_argument("--json", default="runs/competitive_analysis.json")
    args = ap.parse_args()

    print("=" * 104)
    print("  五臂竞品分析")
    print("=" * 104)
    print(f"  {exclusions.summary()}")
    print(f"  成本口径：统一折算价（入 {PRICE_IN} 元/M、出 {PRICE_OUT} 元/M）—— **不是账单**")
    print()

    result: dict = {"arms": ARMS, "configs": {}, "exclusions": exclusions.summary()}
    for label, (run, tkey) in CONFIGS.items():
        data = collect(run, tkey)
        if not data:
            continue
        result["configs"][label] = data
        print("=" * 104)
        print(f"  {label}    run={run}")
        print("=" * 104)
        print(f"  {'臂':<11}{'准确率':>9}{'可判':>7}{'无效':>7}{'截断':>7}"
              f"{'tokens_out':>12}{'中位out':>9}{'中位延迟':>10}{'p95':>8}{'成本(元)':>10}")
        print("  " + "-" * 100)
        for arm in ARMS:
            d = data.get(arm)
            if not d:
                print(f"  {SHORT[arm]:<11}{'（未测）':>9}")
                continue
            print(f"  {SHORT[arm]:<11}"
                  f"{fmt(d.get('acc_lenient'), pct=True):>9}"
                  f"{fmt(d.get('gradeable_lenient')):>7}"
                  f"{fmt(d.get('invalid_lenient'), pct=True):>7}"
                  f"{fmt(d.get('truncated_lenient'), pct=True):>7}"
                  f"{fmt(d.get('tokens_out')):>12}"
                  f"{fmt(d.get('median_tokens_out')):>9}"
                  f"{fmt(d.get('median_latency_s')):>9}s"
                  f"{fmt(d.get('p95_latency_s')):>7}s"
                  f"{fmt(d.get('cost_cny'), nd=4):>10}")
        print()

    stab = stability("c3-5arm-v2")
    if stab:
        result["stability"] = stab
        print("=" * 104)
        print("  稳定性（C3，修正口径：只统计所有 rep 都可解析的题）")
        print("=" * 104)
        print(f"  {'臂':<11}{'修正口径':>10}{'原口径':>10}{'可用题':>8}{'偏差':>9}")
        for arm in ARMS:
            s = stab.get(arm)
            if not s:
                continue
            print(f"  {SHORT[arm]:<11}{s['fixed']:>10.1%}{s['naive']:>10.1%}"
                  f"{s['usable']:>8}{(s['fixed']-s['naive'])*100:>+8.1f}pp")
        print()

    # ---- ★ space-bunny 单独成节（不进主表）--------------------------------
    sb = spacebunny_section()
    if sb:
        result["spacebunny_separate_section"] = sb
        print("=" * 104)
        print("  ★ 单独成节：space-bunny-free（**不进主表**，与其他臂不可横向比较）")
        print("=" * 104)
        print("  原因：它的思考**强制开启**（5 种关思考写法实测全部无效），")
        print("        而主表五臂的思考都被关掉了 ⇒ 「臂」与「思考模式」同时变化。")
        print()
        print(f"  {'题集':<18}{'可判':>6}{'截断率':>9}{'可判项(乐观)':>13}"
              f"{'外推(保守)':>12}{'偏倚':>9}{'主表最好':>10}")
        for label, d in sb.items():
            up = f"{d['acc_upper']:.1%}" if d.get("acc_upper") is not None else "—"
            bp = f"{d['bias_pp']:+.1f}pp" if d.get("bias_pp") is not None else "—"
            bm = f"{d['ref_best_main']:.1%}" if d.get("ref_best_main") is not None else "—"
            print(f"  {label:<18}{d['n_gradeable']:>6}{d['truncated_rate']:>9.1%}"
                  f"{d['acc_gradeable']:>13.1%}{up:>12}{bp:>9}{bm:>10}")
        print()
        print("  读法：区间必须**成对**呈现，并同时报截断率。")
        print("        「主表最好」只是标尺（不同配置），**不可直接相减**。")
        print()

    out = paths.ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
