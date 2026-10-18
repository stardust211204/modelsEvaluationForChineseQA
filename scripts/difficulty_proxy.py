"""难度代理打分：用本地小模型对每题**重复采样**，取通过率作为难度代理。

为什么必须重复采样：
    每题只跑一次只能得到 0/1，无法分「易/中/难」三档。
    重复 k 次得到 k+1 级通过率，再映射到三档才有意义。

为什么用本地模型：
    不占云额度、不产生费用、可断点续跑。我们只用它的**序数**信息
    （"这题相对更难"），不用它的绝对准确率 —— 2B 模型的绝对分数没有意义。

复用统一 ChatClient 的收益：
    * 自动绕过系统代理（本地端点若走代理会被返回 502，见 client.build_http_client）
    * 退避重试与错误分类
    * 同时读 content / reasoning（本地 qwen3.5 也是推理模型，会吃掉 token 预算）

产物：
    data/difficulty_proxy.json       {"<instance_id>": <通过率 0.0~1.0>}
    data/difficulty_proxy_raw.jsonl  原始输出，便于事后复核判分

运行：
    .venv\\Scripts\\python.exe scripts\\difficulty_proxy.py --reps 4
    .venv\\Scripts\\python.exe scripts\\difficulty_proxy.py --reps 4 --n 20   # 小样试跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import paths  # noqa: E402
from meval.client import ChatClient  # noqa: E402
from meval.tally import parse_answer  # noqa: E402
from meval.types import Config, ParseMode, PromptSpec  # noqa: E402

OLLAMA = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3.5:2b"

PROMPT_TMPL = (
    "阅读下面的单项选择题，只回答选项字母，不要解释。\n\n"
    "题目：{q}\n\n{opts}\n\n答案："
)

# 本地 qwen3.5 是推理模型：默认会把 token 预算烧在 thinking 上，
# 导致 content 为空、finish_reason=length（与云端模型同一个坑）。
# 实测 Ollama 的 OpenAI 兼容层对 `think: false` 不稳定，因此靠**放大预算 + 宽松解析**兜底。
GEN_PARAMS = {
    "max_tokens": 2048,
    "temperature": 0.1,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reps", type=int, default=4, help="每题重复次数（决定通过率粒度）")
    ap.add_argument("--n", type=int, default=0, help="0 = 全量")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--base-url", default=OLLAMA)
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    out_p = paths.data() / "difficulty_proxy.json"
    raw_p = paths.data() / "difficulty_proxy_raw.jsonl"

    # 逐题累计命中数（支持续跑：中断后重跑会接着累加）
    hits: dict[str, int] = {}
    tries: dict[str, int] = {}
    if raw_p.exists():
        for line in raw_p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            tries[d["id"]] = tries.get(d["id"], 0) + 1
            hits[d["id"]] = hits.get(d["id"], 0) + int(bool(d.get("hit")))
        if tries:
            print(f"已载入既有采样：{len(tries)} 题，累计 {sum(tries.values())} 次")

    ds = load_dataset("li-lab/MMLU-ProX-Lite", args.lang, split="test")
    total = len(ds) if args.n <= 0 else min(args.n, len(ds))

    client = ChatClient(base_url=args.base_url, api_key="ollama", timeout_s=300, max_retries=2)
    print(f"模型: {args.model}   端点: {args.base_url}")
    print(f"题数: {total}   每题 {args.reps} 次   合计约 {total * args.reps} 次调用\n")

    # 连通性探测
    probe = client.call(PromptSpec(
        run_id="probe", config=Config.C1, arm=args.model,
        instance_id="probe", prompt_text="1+1=?", params={"max_tokens": 64},
    ))
    if not probe.success:
        print(f"!! 本地模型不可用：{probe.error}")
        print("   请确认 Ollama 已启动：ollama list")
        return 2
    print(f"连通性: OK（{probe.latency_s:.1f}s）\n")

    def flush() -> None:
        rates = {k: round(hits[k] / tries[k], 4) for k in tries if tries.get(k)}
        out_p.write_text(json.dumps(rates, ensure_ascii=False), encoding="utf-8")

    t0 = time.perf_counter()
    done_calls = 0
    target_calls = total * args.reps

    with raw_p.open("a", encoding="utf-8") as rawf:
        for i in range(total):
            row = ds[i]
            iid = f"prox-{args.lang}-{row['question_id']}"
            have = tries.get(iid, 0)
            if have >= args.reps:
                continue
            opts_list = [row[f"option_{j}"] for j in range(10) if row.get(f"option_{j}")]
            opts = "\n".join(f"({chr(65 + j)}) {o}" for j, o in enumerate(opts_list))
            prompt = PROMPT_TMPL.format(q=row["question"], opts=opts)

            for _ in range(args.reps - have):
                res = client.call(PromptSpec(
                    run_id="difficulty", config=Config.C1, arm=args.model,
                    instance_id=iid, prompt_text=prompt, params=GEN_PARAMS,
                ))
                if not res.success:
                    print(f"  [{i}] ERR {res.error[:80]}")
                    continue

                # content 可能为空（思考吃满预算）——宽松解析会回退到末行/句式
                idx, step = parse_answer(res.content, len(opts_list), ParseMode.LENIENT)
                hit = idx is not None and idx == int(row["answer_index"])
                tries[iid] = tries.get(iid, 0) + 1
                hits[iid] = hits.get(iid, 0) + int(hit)
                done_calls += 1
                rawf.write(json.dumps({
                    "id": iid, "text": res.content[:200], "step": step,
                    "finish_reason": res.finish_reason,
                    "reasoning_len": len(res.reasoning),
                    "gold": int(row["answer_index"]), "hit": hit,
                }, ensure_ascii=False) + "\n")
                rawf.flush()

                if done_calls % 40 == 0:
                    el = time.perf_counter() - t0
                    rate = done_calls / el if el else 0
                    eta = (target_calls - done_calls) / rate if rate else 0
                    print(f"  {done_calls}/{target_calls} 次  {rate:.1f} 次/秒  ETA {eta/60:.1f} 分钟")
                    flush()

    flush()
    client.close()

    rates = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    if rates:
        vals = sorted(rates.values())
        print(f"\n完成：{len(rates)} 题，每题 {args.reps} 次")
        print(f"  本地模型平均通过率 : {sum(vals)/len(vals):.1%}")
        easy = sum(1 for v in vals if v >= 0.70)
        mid = sum(1 for v in vals if 0.40 <= v < 0.70)
        hard = sum(1 for v in vals if v < 0.40)
        print(f"  难度分档预览       : 易 {easy} / 中 {mid} / 难 {hard}")
        if easy == 0 or mid == 0:
            print("  !! 某一档为空。2B 模型在 10 选项题上随机基线仅 10%，")
            print("     很可能绝大多数题都落在「难」档 —— 难度分层会失去区分度。")
            print("     对策：改用「多模型共识」或提高 reps，或接受两档。")
    print(f"\n  打分 -> {out_p}")
    print(f"  原始 -> {raw_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
