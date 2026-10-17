# -*- coding: utf-8 -*-
"""改 RUN_AUDIT.md / RUN_INVENTORY.md 之前，把要用的数字一次性算准。

产出：
  ① 全库（runs/ + archive/runs）批次与调用总数
  ② A 类 37 个批次的完整表格行（时间按 manifest 重算）
  ③ 三个新失败批次的 error_kind
  ④ 原 15 行的时间修正对照
"""
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RUNS, ARCH = ROOT / "runs", ROOT / "archive" / "runs"
BJ = timezone(timedelta(hours=8))

OLD15 = ["c1-5arm-v2", "c2hi-5arm-v2", "c3-5arm-v2", "c5-5arm-v3", "c6-5arm-v2",
         "ceval-c1-5arm-v2", "ceval-c2hi-5arm-v2", "ceval-c3-5arm",
         "ceval-c5-5arm-v3", "ceval-c5b-5arm", "ceval-c6-5arm",
         "orbench-full", "orbench-judge-full", "orbench-judge-qwen", "orbench-qwen"]

NEW22 = ["c4-5arm", "c4-5arm-nothink",
         "e1-think-none-5arm", "e1-think-high1024-5arm", "e1-think-high8192-5arm",
         "e1b-longcat-think1024", "e1b-longcat-think8192",
         "e2-relabel-5arm", "e2b-relabel-5arm",
         "e6-a-5arm", "e6-b-5arm", "e6-c-5arm", "e6-d-5arm",
         "e5-mmlu-n1-5arm", "e5-mmlu-n3-5arm", "e5-mmlu-n5-5arm", "e5-mmlu-n10-5arm",
         "e5-ceval-n0-5arm", "e5-ceval-n1-5arm", "e5-ceval-n3-5arm",
         "e5-ceval-n5-5arm", "e5-ceval-n10-5arm"]

# A 类里的用途说明（原 15 个沿用 RUN_AUDIT 原文，避免臆改）
USE15 = {
    "c1-5arm-v2": "基线 / MMLU-ProX 588 题",
    "c2hi-5arm-v2": "自由作答 / MMLU-ProX 150 题",
    "c3-5arm-v2": "稳定性 / MMLU-ProX 100×10",
    "c5-5arm-v3": "5-shot / MMLU-ProX 150 题",
    "c6-5arm-v2": "选项乱序 / MMLU-ProX 100×4",
    "ceval-c1-5arm-v2": "基线 / C-Eval 583 题",
    "ceval-c2hi-5arm-v2": "自由作答 / C-Eval 150 题",
    "ceval-c3-5arm": "稳定性 / C-Eval 100×10",
    "ceval-c5-5arm-v3": "5-shot / C-Eval 150 题（用于论证不采用）",
    "ceval-c5b-5arm": "5-shot 对照 / C-Eval 150 题（选项数匹配）",
    "ceval-c6-5arm": "选项乱序 / C-Eval 100×4",
    "orbench-full": "安全 / 源响应 3 臂",
    "orbench-judge-full": "安全 / judge 判定 3 臂",
    "orbench-judge-qwen": "安全 / judge 判定 qwen",
    "orbench-qwen": "安全 / 源响应 qwen",
}
USE22 = {
    "c4-5arm": "judge 复核（思考**未**关，对照）",
    "c4-5arm-nothink": "**judge 复核（思考关）** → 报告 §2.4 五模型一致率",
    "e1-think-none-5arm": "**思考对照·关闭** → 报告 §3.3",
    "e1-think-high1024-5arm": "思考对照·开启（1024 预算，预算不足）",
    "e1-think-high8192-5arm": "**思考对照·开启（8192 预算）** → 报告 §3.3",
    "e1b-longcat-think1024": "longcat 思考开启（1024）",
    "e1b-longcat-think8192": "**longcat 思考开启（8192，可比口径）** → 报告 §3.3",
    "e2-relabel-5arm": "选项标签替换（乱排版 + 破对应）→ 报告 §3.2",
    "e2b-relabel-5arm": "**选项标签替换（仅乱排版）** → 报告 §3.2",
    "e6-a-5arm": "作答格式递进 A（只答字母）→ 报告 §3.3",
    "e6-b-5arm": "递进 B（可写理由，不要求末行）→ 报告 §3.3",
    "e6-c-5arm": "递进 C（要求末行答案）→ 报告 §3.3",
    "e6-d-5arm": "递进 D（再要求逐步推理）→ 报告 §3.3",
    "e5-mmlu-n1-5arm": "5-shot 梯度 n=1（MMLU-ProX）",
    "e5-mmlu-n3-5arm": "5-shot 梯度 n=3",
    "e5-mmlu-n5-5arm": "5-shot 梯度 n=5",
    "e5-mmlu-n10-5arm": "5-shot 梯度 n=10",
    "e5-ceval-n0-5arm": "5-shot 梯度 n=0 基线（C-Eval）",
    "e5-ceval-n1-5arm": "5-shot 梯度 n=1（C-Eval）",
    "e5-ceval-n3-5arm": "5-shot 梯度 n=3",
    "e5-ceval-n5-5arm": "5-shot 梯度 n=5",
    "e5-ceval-n10-5arm": "5-shot 梯度 n=10",
}


def find(rid: str) -> Path | None:
    for base in (RUNS, ARCH):
        if (base / rid).exists():
            return base / rid
    return None


def info(rid: str) -> dict:
    d = find(rid)
    mf_p = d / "manifest.json"
    mf = json.loads(mf_p.read_text(encoding="utf-8")) if mf_p.exists() else {}
    calls = fails = 0
    kinds: Counter = Counter()
    for f in (d / "raw").rglob("*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            calls += 1
            if not r.get("success", True):
                fails += 1
                kinds[r.get("error_kind") or "?"] += 1
    t = mf.get("started_at") or ""
    return {"run_id": rid, "calls": calls, "fails": fails, "kinds": dict(kinds),
            "arms": len(mf.get("arms") or []), "dataset": mf.get("dataset") or "—",
            "config": ",".join(mf.get("configs") or []),
            "inst": len({json.loads(l).get("instance_id")
                         for f in (d / "raw").rglob("*.jsonl")
                         for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
                         if l.strip()}),
            "time": datetime.fromisoformat(t).astimezone(BJ).strftime("%m-%d %H:%M") if t else "—",
            "fp": "、".join(f"{k}={v}" for k, v in (mf.get("prompt_fingerprints") or {}).items())}


print("=" * 118)
print(" ① 全库口径（runs/ + archive/runs）")
print("=" * 118)
tot_d = tot_c = tot_f = 0
for base, tag in ((RUNS, "runs/"), (ARCH, "archive/runs/")):
    ds = sorted(p for p in base.iterdir() if p.is_dir())
    c = f_ = 0
    for d in ds:
        for x in (d / "raw").rglob("*.jsonl"):
            for line in x.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    c += 1
                    if not json.loads(line).get("success", True):
                        f_ += 1
    print(f"  {tag:<16} {len(ds):>3} 个批次目录   {c:>7,} 次调用   失败 {f_}")
    tot_d += len(ds); tot_c += c; tot_f += f_
print(f"  {'合计':<15} {tot_d:>3} 个批次目录   {tot_c:>7,} 次调用   失败 {tot_f}")
print(f"  其中 A 类结论批次：{len(OLD15) + len(NEW22)} 个 —— "
      f"原 15 = {sum(info(r)['calls'] for r in OLD15):,}，新增 22 = {sum(info(r)['calls'] for r in NEW22):,}")

print()
print("=" * 118)
print(" ② RUN_AUDIT.md · A 类新表（37 行，时间按 manifest 重算）")
print("=" * 118)
print("| run_id | 调用 | 失败 | 臂 | 数据集 | 时间 | 说明 |")
print("| --- | ---: | ---: | ---: | --- | --- | --- |")
for rid in OLD15 + NEW22:
    i = info(rid)
    print(f"| `{rid}` | {i['calls']:,} | {i['fails'] or ''} | {i['arms']} | "
          f"{i['dataset']} | {i['time']} | {USE15.get(rid) or USE22[rid]} |")

print()
print("=" * 118)
print(" ③ RUN_INVENTORY.md · §1.4 补充实验（22 行）")
print("=" * 118)
print("| run_id | 配置 | 臂 | 实例数 | 调用 | 北京时间 | prompt 指纹 | 用途 |")
print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
for rid in NEW22:
    i = info(rid)
    use = USE22[rid].replace("→ 报告 ", "")
    print(f"| `{rid}` | {i['config']} | {i['arms']} | {i['inst']} | {i['calls']:,} | "
          f"{i['time']} | `{i['fp']}` | {use} |")

print()
print("=" * 118)
print(" ④ 新增失败批次的 error_kind")
print("=" * 118)
for rid in NEW22:
    i = info(rid)
    if i["fails"]:
        print(f"  {rid:<28} 失败 {i['fails']:>3} / {i['calls']:>4}  "
              f"({i['fails']/i['calls']*100:.1f}%)  {i['kinds']}")

print()
print("=" * 118)
print(" ⑤ 原 15 行的时间修正对照（旧值取自当前 RUN_AUDIT.md）")
print("=" * 118)
OLD_TIME = {"c1-5arm-v2": "10-05 15:31", "c2hi-5arm-v2": "10-05 15:58",
            "c3-5arm-v2": "10-05 16:54", "c5-5arm-v3": "10-07 18:30",
            "c6-5arm-v2": "10-05 17:04", "ceval-c1-5arm-v2": "10-05 15:46",
            "ceval-c2hi-5arm-v2": "10-05 16:06", "ceval-c3-5arm": "10-07 18:17",
            "ceval-c5-5arm-v3": "10-07 18:34", "ceval-c5b-5arm": "10-07 18:42",
            "ceval-c6-5arm": "10-07 18:27", "orbench-full": "10-04 15:36",
            "orbench-judge-full": "10-04 15:50", "orbench-judge-qwen": "10-04 16:59",
            "orbench-qwen": "10-04 16:53"}
for rid in OLD15:
    i = info(rid)
    old = OLD_TIME[rid]
    print(f"  {rid:<22} {old} → {i['time']}   {'✅ 一致' if old == i['time'] else '❌ 修正'}")
