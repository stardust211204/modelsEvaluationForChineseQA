"""三个模型连通性冒烟：N 题 × 各模型，仅作答、关思考、JSONL 落盘、双解析。

产物：data/raw_evidence/smoke_{lang}_{n}.jsonl（探针证据，非主流程产物）
运行：.venv\\Scripts\\python.exe scripts\\smoke.py --n 10 --lang zh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# 统一路径真相来源：scripts/ -> <root> -> src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meval import paths  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ZEN = paths.base_url()
OLLAMA = "http://localhost:11434/v1"
RUN_DIR = paths.evidence()

STRICT_RE = re.compile(r"answer\s*[:：is]*\s*\(?([A-J])\)?", re.IGNORECASE)
BARE_RE = re.compile(r"^\s*\(?([A-J])\)?\s*[.!]?\s*$")
LOOSE_RE = re.compile(r"[A-J]")


def get_key() -> str:
    """统一走 paths.api_key()：环境变量优先，回落用户级环境变量（含 Windows 注册表）。"""
    return paths.api_key()


def build_prompt(row: dict) -> str:
    options = []
    for i in range(10):
        opt = row.get(f"option_{i}")
        if opt:
            options.append(f"({chr(65 + i)}) {opt}")
    return (
        "You are taking a multiple-choice exam. Read the question and options.\n"
        "CRITICAL: Your entire response must be a single capital letter from A to J. "
        "No words, no punctuation, no explanation.\n\n"
        f"Question: {row['question']}\n" + "\n".join(options) + "\n\nAnswer:"
    )


def resolve(text: str) -> tuple[str | None, str | None]:
    """Extraction ladder: bare letter -> 'answer is X' phrase -> lone letter on last lines."""
    if not text:
        return None, None
    t = text.strip()
    m = BARE_RE.match(t)
    if m:
        return m.group(1).upper(), "bare"
    m = STRICT_RE.search(t)
    if m:
        return m.group(1).upper(), "answer-phrase"
    for raw_line in reversed(t.splitlines()[-5:]):
        line = raw_line.replace("*", "").strip()
        m = BARE_RE.match(line)
        if m:
            return m.group(1).upper(), "last-line"
    return None, None


def http_post(url: str, payload: dict, key: str | None, timeout: int = 180):
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "opencode-eval-smoke/0.1",
        "X-Opencode-Session": str(uuid.uuid4()),
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return -1, {"_error": f"{type(e).__name__}: {e}"}


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }


def post_with_retry(url: str, payload: dict, key: str | None):
    status, body = http_post(url, payload, key)
    for attempt in range(2):
        if status == 200:
            return status, body
        if status in (429, 500, 502, 503, 504, -1):
            time.sleep(2 + 3 * attempt)
            status, body = http_post(url, payload, key)
        else:
            break
    return status, body


def responses_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "none"},
    }


def extract_text(protocol: str, body: dict) -> str:
    if protocol == "chat":
        try:
            return body["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001
            return ""
    out = []
    for item in body.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    key = get_key()
    from datasets import load_dataset

    ds = load_dataset("li-lab/MMLU-ProX-Lite", args.lang, split="test")
    rows = [ds[i] for i in range(args.n)]

    arms = [
        dict(id="cloud-deepseek-flash", model="deepseek-flash", protocol="chat", base=ZEN, key=key, max_tokens=512),
        dict(id="cloud-mimo-v2.6-flash", model="mimo-v2.6-flash", protocol="chat", base=ZEN, key=key, max_tokens=512),
        dict(id="cloud-deepseek-v4-flash", model="deepseek-v4-flash", protocol="chat", base=ZEN, key=key, max_tokens=512),
        dict(id="local-qwen3.5-2b", model="qwen3.5:2b", protocol="chat", base=OLLAMA, key=None, max_tokens=512),
    ]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f"smoke_{args.lang}_{args.n}.jsonl"

    summary: dict[str, dict] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for arm in arms:
            stats = summary[arm["id"]] = dict(total=0, ok=0, correct=0, parse_fail=0, bare=0, lat=[], errors=[])
            for row in rows:
                prompt = build_prompt(row)
                if arm["protocol"] == "chat":
                    payload = chat_payload(arm["model"], prompt, arm["max_tokens"])
                    url = f"{arm['base']}/chat/completions"
                else:
                    payload = responses_payload(arm["model"], prompt, arm["max_tokens"])
                    url = f"{arm['base']}/responses"

                t0 = time.perf_counter()
                status, body = post_with_retry(url, payload, arm["key"])
                secs = time.perf_counter() - t0
                text = extract_text(arm["protocol"], body) if status == 200 else ""
                answer, tier = resolve(text) if status == 200 else (None, None)
                correct = bool(answer) and answer == str(row["answer"]).upper()
                rec = dict(
                    ts=datetime.now(timezone.utc).isoformat(),
                    arm=arm["id"],
                    model=arm["model"],
                    protocol=arm["protocol"],
                    lang=args.lang,
                    qid=int(row["question_id"]),
                    prompt=prompt,
                    status=status,
                    text=text,
                    answer=answer,
                    tier=tier,
                    parse_fail=status == 200 and answer is None,
                    correct=correct,
                    latency_s=round(secs, 2),
                    error=body.get("_error") if status != 200 else None,
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["total"] += 1
                if status == 200:
                    stats["ok"] += 1
                    stats["correct"] += int(correct)
                    stats["parse_fail"] += int(rec["parse_fail"])
                    stats["bare"] += int(tier == "bare")
                else:
                    stats["errors"].append(f"HTTP {status}")
            f.flush()

    print(f"\nJSONL -> {out_path}\n")
    print(f"{'arm':<26} {'ok/total':<9} {'acc':<6} {'unresolved':<11} {'bare-rate':<10} errors")
    for arm_id, s in summary.items():
        acc = f"{s['correct'] / s['total']:.0%}" if s["total"] else "-"
        bare = f"{s['bare'] / s['total']:.0%}" if s["total"] else "-"
        errs = ",".join(sorted(set(s["errors"]))) or "-"
        print(f"{arm_id:<26} {s['ok']}/{s['total']:<7} {acc:<6} {s['parse_fail']:<11} {bare:<10} {errs}")


if __name__ == "__main__":
    sys.exit(main())
