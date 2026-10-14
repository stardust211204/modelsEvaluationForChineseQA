"""算出并入补充实验后的「权威批次」调用总数。

原口径（RUN_INVENTORY.md §1）= 39,371 次，含 15 个批次：
    MMLU-ProX  5 个 = 11,440
    C-Eval     6 个 = 12,165
    安全线     4 个 = 15,766

并入的实验（它们现在支撑报告 §2.4 / §2.7 / §3.4 / §3.5 / §3.6 / §3.9 / §5 的结论）
⇒ 按同一口径应计入，总数需更新。
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import paths

# 原来的 15 个权威批次
BASE = {
    "c1-5arm-v2": 2940, "c6-5arm-v2": 2000, "c3-5arm-v2": 5000,
    "c2hi-5arm-v2": 750, "c5-5arm-v3": 750,
    "ceval-c1-5arm-v2": 2915, "ceval-c6-5arm": 2000, "ceval-c3-5arm": 5000,
    "ceval-c2hi-5arm-v2": 750, "ceval-c5-5arm-v3": 750, "ceval-c5b-5arm": 750,
    "orbench-full": 5922, "orbench-qwen": 1974,
    "orbench-judge-full": 5922, "orbench-judge-qwen": 1948,
}

# 本次并入的批次 → 它们支撑报告哪一节
NEW = {
    "e1-think-none-5arm": "§3.9 思考开关（关闭基线）",
    "e1-think-high1024-5arm": "§3.9 思考开关（开启，1024）",
    "e1-think-high8192-5arm": "§3.9 思考开关（开启，8192）",
    "e1b-longcat-think1024": "§3.9 longcat 补跑",
    "e1b-longcat-think8192": "§3.9 longcat 补跑（可比口径）",
    "e2-relabel-5arm": "§3.4 选项标签替换",
    "e2b-relabel-5arm": "§3.4 选项标签替换（排版对照）",
    "c4-5arm": "§2.4 / §3.7 judge 五模型复核",
    "c4-5arm-nothink": "§2.4 / §3.7 judge 复核（思考未关对照）",
    "e5-mmlu-n1-5arm": "§3.6 5-shot 梯度 n=1",
    "e5-mmlu-n3-5arm": "§3.6 5-shot 梯度 n=3",
    "e5-mmlu-n5-5arm": "§3.6 5-shot 梯度 n=5",
    "e5-mmlu-n10-5arm": "§3.6 5-shot 梯度 n=10",
    "e5-ceval-n0-5arm": "§3.6 5-shot 梯度 n=0（基线）",
    "e5-ceval-n1-5arm": "§3.6 5-shot 梯度 n=1",
    "e5-ceval-n3-5arm": "§3.6 5-shot 梯度 n=3",
    "e5-ceval-n5-5arm": "§3.6 5-shot 梯度 n=5",
    "e5-ceval-n10-5arm": "§3.6 5-shot 梯度 n=10",
    "e6-a-5arm": "§3.5 作答格式递进 A",
    "e6-b-5arm": "§3.5 作答格式递进 B",
    "e6-c-5arm": "§3.5 作答格式递进 C",
    "e6-d-5arm": "§3.5 作答格式递进 D",
}


def count(run_id: str) -> tuple[int, int]:
    d = paths.ROOT / "runs" / run_id
    if not d.exists():
        return (-1, -1)
    n = ok = 0
    for fp in d.rglob("*.jsonl"):
        if "raw" not in str(fp):
            continue
        for ln in fp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            n += 1
            if r.get("success"):
                ok += 1
    return (n, ok)


print("=" * 100)
print("  并入的批次")
print("=" * 100)
print(f"  {'run_id':<28}{'调用':>8}{'成功':>8}   用途")
new_n = new_ok = 0
for rid, use in NEW.items():
    n, ok = count(rid)
    if n < 0:
        print(f"  {rid:<28}{'缺失':>8}{'':>8}   {use}")
        continue
    new_n += n
    new_ok += ok
    print(f"  {rid:<28}{n:>8,}{ok:>8,}   {use}")
print(f"  {'小计':<28}{new_n:>8,}{new_ok:>8,}")

print()
print("=" * 100)
print("  总数")
print("=" * 100)
base_n = sum(BASE.values())
print(f"  原 15 个权威批次          {base_n:>8,}")
print(f"  并入的 {len(NEW)} 个批次        {new_n:>8,}")
print(f"  ⇒ 新总数                  {base_n + new_n:>8,}")
print()
print(f"  校验：原口径报告写 39,371，脚本按清单加总得 {base_n:,}"
      f" {'✅ 一致' if base_n == 39371 else '★ 不一致'}")
