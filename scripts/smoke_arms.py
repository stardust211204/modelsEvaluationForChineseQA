"""三个模型连通性冒烟。

目的：在花任何批量额度之前，确认三个模型都能正常作答，且**没有静默空 content**。

「静默空 content」是最危险的一种失败：HTTP 200，但 content 为空串
（推理模型把 token 预算烧在 reasoning_content 上）。
本脚本会明确报告这种情况，而不是把它当成"模型答不出来"。

产物：data/raw_evidence/smoke_arms_{timestamp}.jsonl

运行：
    .venv\\Scripts\\python.exe scripts\\smoke_arms.py
    .venv\\Scripts\\python.exe scripts\\smoke_arms.py --n 3 --lang zh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

import yaml  # noqa: E402

from meval import paths  # noqa: E402
from meval.client import ChatClient  # noqa: E402
from meval.types import Config, PromptSpec  # noqa: E402

PROMPT = (
    "你正在参加一场单项选择题考试。阅读题目与选项，然后作答。\n\n"
    "必须严格遵守：你的整段回复必须且只能是一个大写字母（A 到 J 之一）。\n"
    "不要输出任何单词、标点、解释或空格。\n\n"
    "题目：植物进行光合作用时主要从大气中吸收哪种气体？\n\n"
    "(A) 氧气\n(B) 二氧化碳\n(C) 氮气\n(D) 氢气\n\n答案："
)


def load_arms() -> list[dict]:
    cfg = yaml.safe_load((paths.configs() / "arms.yaml").read_text(encoding="utf-8"))
    return cfg.get("arms", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="每个模型调用次数（默认 3，用于测稳定性）")
    args = ap.parse_args()

    arms = load_arms()
    if not arms:
        print("!! configs/arms.yaml 里没有 arms"); return 2

    print(f"端点: {paths.base_url()}")
    print(f"模型数: {len(arms)}   每个模型 {args.n} 次\n")

    client = ChatClient()
    out_dir = paths.evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"smoke_arms_{stamp}.jsonl"

    rows: list[dict] = []
    verdicts: dict[str, str] = {}

    for arm in arms:
        name = arm["name"]
        params = dict(arm.get("params") or {})
        params.setdefault("max_tokens", 2048)
        params.setdefault("temperature", 0.1)

        print("=" * 74)
        print(f"{name}   params={params}")

        empties = 0
        for i in range(args.n):
            spec = PromptSpec(
                run_id="smoke", config=Config.C1, arm=arm["model"],
                instance_id=f"smoke-{i}", rep=i,
                prompt_text=PROMPT, params=params,
            )
            res = client.call(spec)
            rec = res.to_dict() | {"arm_name": name}
            rows.append(rec)

            flag = "OK " if res.success else "ERR"
            body = res.content[:20].replace("\n", "\\n")
            note = ""
            if res.success and not res.content.strip():
                empties += 1
                note = f"  <<< 空 content! finish={res.finish_reason} reasoning_len={len(res.reasoning)}"
            print(f"  [{flag}] #{i} http={res.http_status} {res.latency_s:>6.2f}s "
                  f"content={body!r} tokens={res.tokens_in}/{res.tokens_out}{note}")
            if res.error:
                print(f"         err: {res.error[:100]}")
            time.sleep(0.3)

        if empties:
            verdicts[name] = f"[WARN] {empties}/{args.n} 次空 content -- 需调大 max_tokens 或换思考关闭参数"
        else:
            verdicts[name] = "[ OK ] 正常"

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    for k, v in verdicts.items():
        print(f"  {k:<28} {v}")
    print(f"\n原始记录 -> {out_path}")

    if any(v.startswith("[WARN]") for v in verdicts.values()):
        print("\n!! 存在空 content：不要进入批量跑批。先跑 probe_params.py 定位原因。")
        return 1
    print("\n可以进入下一步：probe_params.py（逐模型回填生效参数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
