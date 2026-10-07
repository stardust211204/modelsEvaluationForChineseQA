"""端到端集成验证（小样本）：造题 → 构造 → 执行 → 判分。

目的：在花批量额度之前，确认整条管线在新工作区跑得通，
并**实测断点续跑**——第二次运行必须一条都不重复调用。

不依赖 build_set 的完整分层抽样（那需要先跑难度代理）；
这里直接用公开集前 N 题，专测管线连通性。

运行：
    .venv\\Scripts\\python.exe scripts\\smoke_pipeline.py --n 5
    .venv\\Scripts\\python.exe scripts\\smoke_pipeline.py --n 5 --run-id pipe-smoke   # 复跑测续跑
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, execute, paths, tally  # noqa: E402
from meval.types import Config, ParseMode, Tier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="题数")
    ap.add_argument("--run-id", default="pipe-smoke")
    ap.add_argument("--limit", type=int, default=None, help="本次最多执行多少条")
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    print("=" * 76)
    print("端到端集成验证")
    print("=" * 76)

    # ---- ① 造题（简化：直接取公开集前 N 题）------------------------------
    ds = load_dataset("li-lab/MMLU-ProX-Lite", "zh", split="test")
    from meval.types import Instance, Source, Split

    instances = []
    for i in range(args.n):
        r = ds[i]
        opts = [r[f"option_{j}"] for j in range(10) if r.get(f"option_{j}")]
        instances.append(Instance(
            id=f"prox-zh-{r['question_id']}", question=r["question"], options=opts,
            gold_indices=[int(r["answer_index"])], split=Split.TEST, source=Source.PUBLIC,
            stratum={"subject": str(r.get("category", "unknown"))},
        ))
    print(f"① 造题      : {len(instances)} 题（简化，未做分层抽样）")

    # ---- ② 构造 -----------------------------------------------------------
    _cfgall, arms = adapt.load_arms()
    c1 = next(c for c in adapt.load_matrix() if c.key is Config.C1)
    tmpl = (paths.ROOT / c1.prompt_template).read_text(encoding="utf-8")
    specs = adapt.build_specs(
        run_id=args.run_id, config=c1, arms=arms, instances=instances, prompt_template=tmpl,
    )
    print(f"② 构造      : {len(specs)} 条 PromptSpec（{len(arms)} 臂 × {len(instances)} 题）")
    assert len(specs) == len(arms) * len(instances)

    # 纯函数自检
    specs2 = adapt.build_specs(
        run_id=args.run_id, config=c1, arms=arms, instances=instances, prompt_template=tmpl,
    )
    assert [s.to_dict() for s in specs] == [s.to_dict() for s in specs2], "adapt 不是纯函数！"
    print("   纯函数自检 : 通过（两次构造 byte-identical）")

    # ---- ③ 执行（跑两遍，第二遍验证续跑）----------------------------------
    conc = {a.model: a.concurrency for a in arms}
    print(f"③ 执行      : 并发设置 {conc}")
    stats1 = execute.run_specs(
        run_id=args.run_id, specs=specs, arms_concurrency=conc, limit=args.limit,
    )
    print(f"   第一遍    : 执行 {stats1['executed']}｜失败 {stats1['failed']}｜"
          f"跳过 {stats1['skipped_done']}")

    stats2 = execute.run_specs(
        run_id=args.run_id, specs=specs, arms_concurrency=conc, limit=args.limit,
    )
    print(f"   第二遍    : 执行 {stats2['executed']}｜失败 {stats2['failed']}｜"
          f"跳过 {stats2['skipped_done']}")
    resume_ok = int(stats2["executed"]) == 0
    print(f"   续跑验证  : {'通过（零重复调用）' if resume_ok else '失败（重复调用了！）'}")

    # ---- ④ 判分 -----------------------------------------------------------
    results = execute.load_results(args.run_id, Config.C1)
    by_id = {i.id: i for i in instances}
    tally_rows = []
    for r in results:
        inst = by_id.get(r.instance_id)
        if inst is None:
            continue
        for t in tally.tally_both_modes(r, inst):
            tally_rows.append(t)

    print(f"④ 判分      : {len(results)} 条结果 -> {len(tally_rows)} 条 Tally（双口径）")

    from collections import Counter

    strict = Counter(t.tier.value for t in tally_rows if t.mode is ParseMode.STRICT)
    lenient = Counter(t.tier.value for t in tally_rows if t.mode is ParseMode.LENIENT)
    print(f"   严格口径  : {dict(strict)}")
    print(f"   宽松口径  : {dict(lenient)}")

    n_trunc = sum(1 for t in tally_rows if t.tier is Tier.TRUNCATED)
    if n_trunc:
        print(f"   !! 出现 {n_trunc} 条 truncated —— 检查 max_tokens 与思考开关")

    # ---- 落盘 Tally -------------------------------------------------------
    out_dir = paths.tallies_dir(args.run_id, Config.C1.value)
    p = out_dir / "smoke_tally.jsonl"
    p.write_text("\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in tally_rows) + "\n",
                 encoding="utf-8")

    # ---- manifest ---------------------------------------------------------
    mp = execute.write_manifest(
        run_id=args.run_id, configs=[Config.C1.value],
        arms=[a.to_dict() for a in arms],
        data_hash="smoke-" + str(len(instances)),
        models_snapshot=[a.model for a in arms],
        note="端到端集成验证（小样本，未做分层抽样）", stats=stats1,
    )
    print(f"⑤ 元数据    : {mp.relative_to(paths.ROOT)}")
    print(f"   Tally     : {p.relative_to(paths.ROOT)}")

    print("\n" + "=" * 76)
    ok = resume_ok and not n_trunc and int(stats1["failed"]) == 0
    print("结论：" + ("全部通过，可以进入正式造题与批量跑批" if ok else "存在问题，见上方标记"))
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
