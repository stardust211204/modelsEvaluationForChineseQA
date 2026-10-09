"""逐个核对主评测线图上的每个环节是否真实存在于代码里。

图上（pipeline_main_v9.tex）声称的流程：
    数据源 →[抽样]→ benchmark →[prompt 生成]→ 五条作答路径 →[模型推理]→ raw
           →[规则判分]→ 判定结果 →[得出结论]→ 指标
           →[LLM judge 复核]──────────────────→ 指标

本脚本把每个环节落到**具体的文件与行**，并给出该环节的输入输出类型，
确认它不是画图时想当然加上去的。

用法：.venv\\Scripts\\python.exe scripts\\_verify_pipeline_steps.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 环节 → 在哪些文件里找什么证据
CHECKS = [
    ("抽样", ["src/meval/build_set.py"],
     [r"def\s+(stratified|sample|select|build_\w+)", r"分层|抽样|间隔"],
     "从题池按固定间隔/分层抽取子集"),
    ("prompt 生成", ["src/meval/adapt.py"],
     [r"def\s+render_prompt", r"def\s+render_options", r"def\s+render_demos",
      r"def\s+build_specs", r"def\s+prompt_fingerprint"],
     "把实例渲染成提示词，并生成 PromptSpec"),
    ("模型推理", ["src/meval/execute.py", "src/meval/client.py"],
     [r"def\s+run_specs", r"def\s+(call|chat|post|complete)", r"concurrency|并发"],
     "按 spec 调 API，把回复写进 raw"),
    ("规则判分", ["src/meval/tally.py", "src/meval/metrics.py"],
     [r"def\s+(grade|classify|tier|parse)\w*", r"STRICT|LENIENT|strict|lenient"],
     "解析回复、判 tier、写入 tally"),
    ("LLM judge 复核", ["scripts/run_judge.py", "scripts/run_refusal_judge.py"],
     [r"judge|JUDGE"],
     "用 ds-pro 复核规则判分"),
    ("得出结论 / 指标", ["src/meval/metrics.py"],
     [r"def\s+\w*(accuracy|consistency|mcnemar|loo|refusal)\w*"],
     "聚合出各维度指标"),
]

# 图上的节点是否真有对应的数据文件
ARTIFACTS = [
    ("benchmark（题集）", ["data/eval_set.jsonl", "data/ceval_set.jsonl"]),
    ("raw（原始回复）", ["runs/c1-5arm-v2/raw"]),
    ("判定结果（tally）", ["runs/c1-5arm-v2/tallies"]),
    ("指标（图件/汇总）", ["figures", "runs/c4-full"]),
]


def main() -> int:
    print("=" * 96)
    print("  主评测线：图上每个环节 → 代码里的落点")
    print("=" * 96)
    ok = 0
    bad = []
    for step, files, pats, desc in CHECKS:
        print(f"\n【{step}】{desc}")
        found_any = False
        for rel in files:
            p = ROOT / rel
            if not p.exists():
                print(f"    ✗ {rel} 不存在")
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            hits = []
            for pat in pats:
                for m in re.finditer(pat, text):
                    line = text[: m.start()].count("\n") + 1
                    hits.append((line, text.splitlines()[line - 1].strip()[:76]))
            if hits:
                found_any = True
                print(f"    ✓ {rel}")
                for line, src in hits[:4]:
                    print(f"        L{line}: {src}")
            else:
                print(f"    ✗ {rel} 里没找到匹配")
        if found_any:
            ok += 1
        else:
            bad.append(step)

    print("\n" + "=" * 96)
    print("  图上的节点 → 磁盘上的产物")
    print("=" * 96)
    for name, paths in ARTIFACTS:
        for rel in paths:
            p = ROOT / rel
            if p.exists():
                n = len(list(p.rglob("*"))) if p.is_dir() else 1
                print(f"    ✓ {name:<22} {rel}  ({'目录' if p.is_dir() else '文件'}，{n} 项)")
            else:
                print(f"    ✗ {name:<22} {rel}  不存在")

    print("\n" + "=" * 96)
    print(f"  环节核对：{ok} / {len(CHECKS)} 个在代码里找到落点")
    if bad:
        print(f"  ★ 没找到的：{', '.join(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
