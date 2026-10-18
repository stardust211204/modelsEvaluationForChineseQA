"""三个模型互异性指纹：确认三个模型 ID 背后确实是三个不同的模型。

背景：`resp_model` 字段只是回显请求名，不含后端信息（实测确认）。
因此只能用**行为证据** —— 同一批题，两个真别名的答案应几乎完全一致。

⚠️ 解读前必须先剔除空响应。曾出现三个模型 93–100% 一致的假象，
   全部由共同返回空 content 造成（见 docs/param_matrix.md §6）。

产物：data/raw_evidence/probe_distinct.jsonl
运行：.venv\\Scripts\\python.exe scripts\\probe_distinct.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

import yaml  # noqa: E402

from meval import paths  # noqa: E402
from meval.client import ChatClient  # noqa: E402
from meval.tally import parse_answer  # noqa: E402
from meval.types import Config, ParseMode, PromptSpec  # noqa: E402

N_Q = 30

PROMPT_TMPL = (
    "你正在参加一场单项选择题考试。阅读题目与选项，然后作答。\n\n"
    "必须严格遵守：你的整段回复必须且只能是一个大写字母（A 到 J 之一）。\n"
    "不要输出任何单词、标点、解释或空格。\n\n"
    "题目：{q}\n\n{opts}\n\n答案："
)


def load_arms() -> list[dict]:
    cfg = yaml.safe_load((paths.configs() / "arms.yaml").read_text(encoding="utf-8"))
    return cfg.get("arms", [])


def main() -> int:
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    ds = load_dataset("li-lab/MMLU-ProX-Lite", "zh", split="test")
    rows = [ds[i] for i in range(N_Q)]

    arms = load_arms()
    client = ChatClient()
    records: list[dict] = []
    answers: dict[str, list[str | None]] = {}

    print(f"指纹题数: {N_Q}   模型数: {len(arms)}\n")

    for arm in arms:
        name, model = arm["name"], arm["model"]
        params = dict(arm.get("params") or {})
        got: list[str | None] = []
        empties = 0
        for row in rows:
            opts = "\n".join(
                f"({chr(65 + i)}) {row[f'option_{i}']}"
                for i in range(10) if row.get(f"option_{i}")
            )
            prompt = PROMPT_TMPL.format(q=row["question"], opts=opts)
            n_opts = sum(1 for i in range(10) if row.get(f"option_{i}"))
            spec = PromptSpec(
                run_id="distinct", config=Config.C1, arm=model,
                instance_id=f"q{row['question_id']}", prompt_text=prompt, params=params,
            )
            res = client.call(spec)
            records.append(res.to_dict() | {"arm_name": name, "qid": int(row["question_id"])})
            idx, _ = parse_answer(res.content, n_opts, ParseMode.LENIENT)
            if not res.success or idx is None:
                empties += 1
                got.append(None)
            else:
                got.append(chr(65 + idx))
        answers[name] = got
        print(f"  {name:<28} 有效作答 {N_Q - empties}/{N_Q}"
              + (f"  ({empties} 条空/不可解析)" if empties else ""))

    print("\n" + "=" * 78)
    print("两两完全一致率（仅统计双方都有有效答案的题）")
    print("=" * 78)
    names = list(answers)
    verdict: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = answers[names[i]], answers[names[j]]
            both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if not both:
                print(f"  {names[i]:<26} vs {names[j]:<26} 无可比题")
                continue
            same = sum(1 for x, y in both if x == y)
            pct = same / len(both) * 100
            mark = "  <<< 可疑：疑似同一模型" if pct >= 80 else ""
            print(f"  {names[i]:<26} vs {names[j]:<26} {same:>3}/{len(both):<3} {pct:>5.1f}%{mark}")
            if pct >= 80:
                verdict.append(f"{names[i]} vs {names[j]} = {pct:.0f}%")

    out = paths.evidence() / "probe_distinct.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    if verdict:
        print("!! 发现高度一致的对，疑似别名：")
        for v in verdict:
            print("   " + v)
        print("   -> 需换模型，或说明这两个模型为何仍可比较")
    else:
        print("三个模型互异性通过：无一对达到 80% 一致阈值")
    print(f"\n原始记录 -> {out}  ({len(records)} 次调用)")
    client.close()
    return 1 if verdict else 0


if __name__ == "__main__":
    raise SystemExit(main())
