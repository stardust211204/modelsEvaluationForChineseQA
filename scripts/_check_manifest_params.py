# -*- coding: utf-8 -*-
"""核对 manifest.params_effective 与 raw 记录里的实际请求参数是否一致。

背景：c4-5arm-nothink 的 manifest 漏记了 `reasoning_effort: "none"`，
      而 raw 记录里是对的。params_effective 是 §4.4 参数分组与 ci_gate 的依据，
      所以要查清是个例还是普遍问题。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]

bad, ok, nosrc = [], 0, []
for base in (ROOT / "runs", ROOT / "archive" / "runs"):
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        mfp = d / "manifest.json"
        if not mfp.exists():
            nosrc.append(d.name)
            continue
        pe = json.loads(mfp.read_text(encoding="utf-8")).get("params_effective") or {}
        # 从 raw 里归纳每个 arm 的真实参数
        real: dict[str, Counter] = defaultdict(Counter)
        for f in (d / "raw").rglob("*.jsonl"):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("success", True):
                    continue
                real[str(r.get("arm"))][json.dumps(r.get("params") or {},
                                                   ensure_ascii=False, sort_keys=True)] += 1
        for arm, cnt in real.items():
            top = json.loads(cnt.most_common(1)[0][0])
            declared = pe.get(arm)
            if declared is None:
                bad.append((d.name, arm, "manifest 无该 arm", top))
            elif json.dumps(declared, sort_keys=True) != json.dumps(top, sort_keys=True):
                bad.append((d.name, arm, json.dumps(declared, ensure_ascii=False, sort_keys=True),
                            json.dumps(top, ensure_ascii=False, sort_keys=True)))
            else:
                ok += 1

print(f"逐 arm 核对：一致 {ok} 处｜**不一致 {len(bad)} 处**｜无 manifest 的批次 {len(nosrc)} 个\n")
if bad:
    print("=" * 110)
    print(" 不一致明细（manifest.params_effective  vs  raw 记录的实际请求参数）")
    print("=" * 110)
    for rid, arm, dec, real in bad:
        print(f"\n  【{rid}】 arm={arm}")
        print(f"     manifest : {dec}")
        print(f"     raw 实际 : {real}")
if nosrc:
    print(f"\n无 manifest（跳过）：{', '.join(nosrc)}")
