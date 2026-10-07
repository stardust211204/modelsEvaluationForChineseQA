"""C5/C-Eval 异常归因：选项数错配是不是 −13.3 百分点的主因？

背景
    C5（5-shot）在两个评测集上共用同一组示例，示例选项数 10/10/10/8/4（均值 8.4），
    而 C-Eval 全部 583 题都是 4 选项。实测 C5/C-Eval 上 ds-pro 相对 ds-flash
    掉 13.3 个百分点，而 MMLU 上只有 0.7。

    已排除：判分失败（150 题全部可判，无效与截断均为 0）。
    已定位：ds-pro 的位置偏好被放大（标准答案为 B 的题上错选 D 由 17.6% 升到 40.5%）。
    待检验：**示例的选项数与目标题错配**是不是触发这个放大的原因。

本轮补跑提供的对照组
    c5-5arm-v3       / ceval-c5-5arm-v3      示例均值 8.4（清洗 N/A 后重跑，作基线）
    c5b-5arm         / ceval-c5b-5arm        示例均值 10.0（MMLU）/ 5.0（C-Eval）

判读
    若 C-Eval 上 c5b 的 ds-pro 明显回升 ⇒ 选项数错配是主因
    若 c5b 与 c5-v3 无差别            ⇒ 错配不是主因，位置偏好是模型固有特性
"""

import json
import sys
from collections import defaultdict, Counter

sys.path.insert(0, "src")
from meval import build_set, exclusions, metrics, paths  # noqa: E402
from meval.types import Config  # noqa: E402

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
SH = {"deepseek-v4.1-flash": "ds-flash", "deepseek-v4-pro": "ds-pro",
      "longcat-2.5-preview-free": "longcat", "mimo-v2.6-flash": "mimo",
      "qwen3.8-flash": "qwen"}
DROP = exclusions.load()


def load(run, cfg, mode="strict"):
    ts = metrics.load_tallies(run, Config(cfg))
    return [t for t in ts if t["mode"] == mode and t["instance_id"] not in DROP]


def acc(rows, arm):
    v = [t["tier"] for t in rows if t["arm"] == arm and t["tier"] in ("correct", "wrong")]
    return sum(1 for x in v if x == "correct") / len(v) * 100 if v else float("nan")


def picks(rows, arm, gold_filter=None, pool=None):
    """标准答案满足 gold_filter 时，该臂选了哪些字母。"""
    c = Counter()
    for t in rows:
        if t["arm"] != arm:
            continue
        inst = pool.get(t["instance_id"])
        if inst is None:
            continue
        g = "".join(chr(65 + x) for x in inst.gold_indices)
        if gold_filter and g != gold_filter:
            continue
        ai = t.get("answer_index")
        c[chr(65 + ai) if ai is not None else "未解析"] += 1
    return c


def main() -> int:
    ce = {i.id: i for i in build_set.load_frozen(paths.data() / "ceval_set.jsonl")}

    print("=" * 92)
    print("  C5/C-Eval 异常归因（选项数错配 vs 模型固有位置偏好）")
    print("=" * 92)

    # ---- 对照组是否就绪 ----
    groups = [
        ("C1 基线 / C-Eval",     "ceval-c1-5arm-v2",   "c1"),
        ("C5 旧（示例均值 8.4）",  "ceval-c5-5arm",      "c5"),
        ("C5 重跑（清洗后 8.4）",  "ceval-c5-5arm-v3",   "c5"),
        ("C5b（示例均值 5.0）",    "ceval-c5b-5arm",     "c5b"),
    ]
    ready = []
    for label, run, cfg in groups:
        p = paths.ROOT / "runs" / run / "manifest.json"
        if p.exists():
            ready.append((label, run, cfg))
        else:
            print(f"  ⏳ 尚未就绪：{label}  ({run})")
    if not ready:
        print("\n  没有可分析的批次 —— 等补跑完成后再运行。")
        return 0

    print(f"\n  可分析 {len(ready)} 组\n")

    # ---- ① 总体准确率对照 ----
    print("  ① 各组的严格口径准确率（C-Eval，已过滤排除清单）")
    print(f"  {'组':<24}" + "".join(f"{SH[a]:>10}" for a in ARMS))
    print("  " + "-" * 74)
    for label, run, cfg in ready:
        rows = load(run, cfg)
        line = f"  {label:<24}"
        for a in ARMS:
            v = acc(rows, a)
            line += f"{v:>9.1f}%" if v == v else f"{'—':>10}"
        print(line)

    # ---- ② 旗舰 vs flash 差距 ----
    print("\n  ② ds-pro 相对 ds-flash 的差距（百分点，负 = 旗舰更低）")
    print(f"  {'组':<24}{'ds-flash':>10}{'ds-pro':>10}{'差距':>10}")
    print("  " + "-" * 56)
    for label, run, cfg in ready:
        rows = load(run, cfg)
        f, p = acc(rows, "deepseek-v4.1-flash"), acc(rows, "deepseek-v4-pro")
        print(f"  {label:<24}{f:>9.1f}%{p:>9.1f}%{p - f:>+10.1f}")

    # ---- ③ 位置偏好：选 D 的比例 ----
    print("\n  ③ ds-pro 的答案字母分布（位置偏好的直接证据）")
    print(f"  {'组':<24}{'A':>6}{'B':>6}{'C':>6}{'D':>6}   D 占比")
    print("  " + "-" * 62)
    for label, run, cfg in ready:
        rows = load(run, cfg)
        c = picks(rows, "deepseek-v4-pro", pool=ce)
        tot = sum(c.values()) or 1
        print(f"  {label:<24}"
              + "".join(f"{c.get(k, 0):>6}" for k in "ABCD")
              + f"   {c.get('D', 0) / tot * 100:>5.1f}%")

    # ---- ④ 标准答案为 B 时错选 D 的比例（核心指标）----
    print("\n  ④ 标准答案为 B 的实例上，错选 D 的比例（核心指标）")
    print(f"  {'组':<24}{'ds-flash':>11}{'ds-pro':>11}{'其余三臂':>12}")
    print("  " + "-" * 60)
    for label, run, cfg in ready:
        rows = load(run, cfg)
        cell = []
        for a in ("deepseek-v4.1-flash", "deepseek-v4-pro"):
            c = picks(rows, a, gold_filter="B", pool=ce)
            tot = sum(c.values()) or 1
            cell.append(c.get("D", 0) / tot * 100)
        others = []
        for a in ("longcat-2.5-preview-free", "mimo-v2.6-flash", "qwen3.8-flash"):
            c = picks(rows, a, gold_filter="B", pool=ce)
            tot = sum(c.values()) or 1
            others.append(c.get("D", 0) / tot * 100)
        print(f"  {label:<24}{cell[0]:>10.1f}%{cell[1]:>10.1f}%{sum(others)/len(others):>11.1f}%")

    # ---- 结论提示 ----
    print("\n  判读")
    print("    若「C5b（示例均值 5.0）」的 ds-pro 明显高于「C5 重跑」")
    print("      ⇒ 示例与目标题的选项数错配是主因，可通过改示例修复")
    print("    若两组无差别")
    print("      ⇒ 错配不是主因；ds-pro 的位置偏好是模型固有特性，")
    print("        报告应把该异常描述为「模型特性」而非「设置缺陷」")

    dst = paths.ROOT / "reports" / "c5_ceval_attribution.json"
    out = {}
    for label, run, cfg in ready:
        rows = load(run, cfg)
        cd = picks(rows, "deepseek-v4-pro", pool=ce)
        cfb = picks(rows, "deepseek-v4-pro", gold_filter="B", pool=ce)
        out[label] = {
            "run": run, "config": cfg,
            "acc": {SH[a]: round(acc(rows, a), 2) for a in ARMS},
            "dspro_gap_vs_dsflash": round(acc(rows, "deepseek-v4-pro")
                                          - acc(rows, "deepseek-v4.1-flash"), 2),
            "dspro_letter_picks": dict(cd),
            "dspro_pick_D_rate": round(cd.get("D", 0) / (sum(cd.values()) or 1) * 100, 2),
            "dspro_pick_D_when_gold_B": round(cfb.get("D", 0) / (sum(cfb.values()) or 1) * 100, 2),
        }
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已写出 {dst.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
