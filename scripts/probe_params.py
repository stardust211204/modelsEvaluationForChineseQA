"""F0 probe: does the aggregator pool honor sampling parameters?

Answers three questions with evidence rather than assumption:
  1. What models does the pool expose?          -> GET /models
  2. Which params are ACCEPTED vs IGNORED?      -> send variants, inspect HTTP status + echoed body
  3. Is `temperature` actually EFFECTIVE?       -> same question x N calls at temp=1.0
                                                   (if outputs are byte-identical, it is being ignored)

Writes data/raw_evidence/probe_params.jsonl and prints a report. Read-only w.r.t. the pool.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))
from smoke import ZEN, get_key, http_post  # noqa: E402
from meval import paths  # noqa: E402

OUT = paths.evidence()
QUESTION = (
    "Which gas do plants primarily absorb from the atmosphere for photosynthesis?\n"
    "(A) Oxygen\n(B) Carbon dioxide\n(C) Nitrogen\n(D) Hydrogen\n\nAnswer:"
)


def list_models(key: str) -> list[str]:
    req = urllib.request.Request(
        f"{ZEN}/models",
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "opencode-eval-probe/0.1",
            "X-Opencode-Session": str(uuid.uuid4()),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        data = body.get("data") or body.get("models") or []
        return [m.get("id") or m.get("name") for m in data if isinstance(m, dict)]
    except Exception as e:  # noqa: BLE001
        print(f"  !! /models failed: {type(e).__name__}: {e}")
        return []


def call(model: str, key: str, prompt: str, **params) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 64}
    payload.update(params)
    t0 = time.perf_counter()
    status, body = http_post(f"{ZEN}/chat/completions", payload, key)
    dt = time.perf_counter() - t0
    text = ""
    if status == 200:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001
            text = "<unparsable body>"
    return {
        "model": model, "params": params, "status": status, "secs": round(dt, 2),
        "resp_model": (body.get("model") if isinstance(body, dict) else None),
        "text": text[:200],
        "error": (body.get("_error") if isinstance(body, dict) and status != 200 else None),
    }


def main() -> None:
    key = get_key()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print("=" * 78)
    print("1) POOL INVENTORY  GET /models")
    print("=" * 78)
    models = list_models(key)
    if models:
        for m in models:
            print(f"   - {m}")
    else:
        print("   (no models returned; falling back to previously used ids)")
        models = ["deepseek-flash", "mimo-v2.6-flash"]

    # limit to a few arms to keep the probe cheap
    probe_models = models[:4] if len(models) >= 4 else models
    print(f"\n   probing {len(probe_models)} model(s): {', '.join(probe_models)}")

    print()
    print("=" * 78)
    print("2) PARAM ACCEPTANCE  which params survive?")
    print("=" * 78)
    variants = {
        "bare":          {},
        "temp0":         {"temperature": 0},
        "temp0.1":       {"temperature": 0.1},
        "temp1":         {"temperature": 1.0},
        "tp0.9":         {"temperature": 0.1, "top_p": 0.9},
        "seed42":        {"temperature": 0.1, "seed": 42},
        "thinking_off":  {"temperature": 0.1, "thinking": {"type": "disabled"}},
        "re_none":       {"temperature": 0.1, "reasoning_effort": "none"},
    }
    for m in probe_models:
        print(f"\n   --- {m} ---")
        for name, params in variants.items():
            r = call(m, key, QUESTION, **params)
            r["variant"] = name
            rows.append(r)
            flag = "OK " if r["status"] == 200 else "ERR"
            echo = f"resp_model={r['resp_model']}" if r["resp_model"] else ""
            err = (r["error"] or "")[:90].replace("\n", " ")
            print(f"     [{flag}] {name:<12} http={r['status']:<4} {r['secs']:>5.2f}s  "
                  f"text={r['text'][:24]!r} {echo} {err}")

    print()
    print("=" * 78)
    print("3) IS temperature EFFECTIVE?  same question x 6 at temperature=1.0")
    print("=" * 78)
    for m in probe_models:
        outs = []
        for i in range(6):
            r = call(m, key, QUESTION, temperature=1.0)
            r["variant"] = f"temp1_rep{i}"
            rows.append(r)
            outs.append(r["text"])
        uniq = len(set(outs))
        verdict = "IGNORED (deterministic)" if uniq == 1 else f"EFFECTIVE ({uniq} distinct)"
        print(f"   {m:<26} distinct outputs = {uniq}/6  -> temp is {verdict}")

    with (OUT / "param_probe.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n   raw -> {OUT / 'param_probe.jsonl'}  ({len(rows)} calls)")


if __name__ == "__main__":
    main()
