"""C7 · 拒答/弃权评测（**双向**）。

为什么必须双向
    只测一侧一定会得出错误结论：
      * 只测「该拒未拒」→ 越保守的模型越好看，会奖励"什么都拒"的模型。
      * 只测「过度拒答」→ 越放开的模型越好看，会奖励"什么都答"的模型。
    安全评测的标准做法是**同时报两侧**，把二者当权衡而不是当分数。

与准确率评测的关系
    完全独立。拒答判的是「该不该答」，准确率判的是「答得对不对」。
    两侧共用同一套 Tally 结构、同一批模型、同一套 raw/tally 分离纪律，
    但**档位与分母完全不同**（见 metrics.refusal_accounting）。

纪律
    `truncated` 只归因于**我们的配置**，绝不混进任何"安全"分母 ——
    否则"我把 token 设小了"会被读成"模型不安全"。
    `partial`（加了免责但给了实质内容）一律人工复核。

用法
    .venv\\Scripts\\python.exe scripts\\run_refusal.py --limit 6      # 试跑
    .venv\\Scripts\\python.exe scripts\\run_refusal.py               # 全量
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, PromptSpec  # noqa: E402

RAW_FILES = ["refusal_over_raw.json", "refusal_under_raw.json"]
FROZEN = "refusal_set.jsonl"


def load_items(items_file: str | None = None) -> list[dict]:
    """加载拒答题集。

    默认合并自建的两侧中文题集（分文件存放：两侧**审校标准完全不同**，
    混在一个文件里容易串味）。传入 `items_file` 时改读单个 jsonl ——
    用于 OR-Bench 等外部题集（见 `scripts/build_orbench_set.py`）。
    """
    if items_file:
        p = Path(items_file)
        if not p.is_absolute():
            p = paths.ROOT / p
        if not p.exists():
            raise SystemExit(f"!! 未找到 {p}")
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    else:
        items = []
        for name in RAW_FILES:
            p = paths.data() / name
            if not p.exists():
                raise SystemExit(f"!! 未找到 {p}")
            items.extend(json.loads(p.read_text(encoding="utf-8"))["items"])

    ids = Counter(i["id"] for i in items)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"!! id 重复：{dup[:5]}（会互相覆盖 tally）")

    missing = [i.get("id") for i in items
               if not i.get("prompt") or "should_refuse" not in i]
    if missing:
        raise SystemExit(f"!! 条目缺字段：{missing[:5]}")

    sides = Counter("under" if i["should_refuse"] else "over" for i in items)
    if not sides.get("over") or not sides.get("under"):
        raise SystemExit(f"!! 两侧必须都有条目（当前 {dict(sides)}）—— 单侧无法测权衡")
    return items


def freeze(items: list[dict]) -> str:
    """冻结成 jsonl 并返回哈希 —— 与题集同一套纪律：产出后不得再改。"""
    import hashlib

    payload = "\n".join(json.dumps(i, ensure_ascii=False, sort_keys=True) for i in items)
    (paths.data() / FROZEN).write_text(payload + "\n", encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="C7 拒答评测")
    ap.add_argument("--run-id", default="c7-full")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--items-file", default=None, help="改用外部题集 jsonl（如 OR-Bench）")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="★ 拒答回复比 MCQ 长得多。实测 max_tokens=1024 时 "
                         "25%% 被截断（tokens_out 中位已达 741），截断会让档位判定不可靠")
    args = ap.parse_args()

    items = load_items(args.items_file)
    h = freeze(items)
    _cfg, arms = adapt.load_arms()
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",")}
        arms = [a for a in arms if a.name in keep]
    by_id = {i["id"]: i for i in items}

    n_over = sum(1 for i in items if not i["should_refuse"])
    n_under = sum(1 for i in items if i["should_refuse"])
    print("=" * 78)
    print(f"C7 拒答评测   run_id={args.run_id}")
    print("=" * 78)
    print(f"  种子集      : {len(items)} 条（过度拒答探针 {n_over} + 应拒探针 {n_under}）  hash={h}")
    print(f"  模型          : {', '.join(a.name for a in arms)}")
    print(f"  待执行      : {len(items) * len(arms)} 条")

    if not args.report_only:
        specs = [
            PromptSpec(run_id=args.run_id, config=Config.C7, arm=a.model,
                       instance_id=it["id"], rep=0,
                       prompt_text=it["prompt"],
                       # ★ 覆盖 max_tokens：拒答回复远长于 MCQ 的裸字母，
                       #   用 arms.yaml 的 1024 会让 25% 的回复被截断。
                       params={**a.params, "max_tokens": args.max_tokens})
            for it in items for a in arms
        ]
        stats = execute.run_specs(
            run_id=args.run_id, specs=specs,
            arms_concurrency={a.model: a.concurrency for a in arms}, limit=args.limit,
        )
        print(f"  失败        : {stats.get('failed')}")
        # ★ configs 留空：C7 不在 matrix.yaml 里（它用的是自己的题集，不是 C1 子集）。
        #   dataset="refusal_set" 让门禁知道**别拿 eval_set 的哈希来比** ——
        #   否则会把"用了另一份数据集"误报成"题集被改了、此 run 不可配对"。
        execute.write_manifest(
            run_id=args.run_id, configs=[],
            arms=[{"name": a.name, "params": a.params} for a in arms],
            data_hash=h, dataset=("or-bench" if args.items_file else "refusal_set"),
            models_snapshot=[a.model for a in arms],
            note=f"C7 拒答评测（双向）｜题集 {len(items)} 条"
                 f"（过度拒答 {n_over} + 应拒 {n_under}）｜{len(items)*len(arms)} 条请求",
            stats=stats,
        )

    results = execute.load_results(args.run_id, Config.C7)
    if not results:
        print("!! 没有结果")
        return 2

    rows = [tally.tally_refusal(r, by_id[r.instance_id]["should_refuse"], Config.C7)
            for r in results if r.instance_id in by_id]
    tp = paths.tallies_dir(args.run_id, Config.C7.value) / "all.jsonl"
    tp.write_text("\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in rows) + "\n",
                  encoding="utf-8")
    trows = [t.to_dict() for t in rows]

    report: dict = {"run_id": args.run_id, "n_items": len(items),
                    "set_hash": h, "arms": {}}
    print()
    print("=" * 78)
    print("拒答分账（两侧必须同时看）")
    print("=" * 78)
    for side, label, want in (("over", "不该拒（过度拒答侧）", False),
                              ("under", "该拒（危险服从侧）", True)):
        ids = {i["id"] for i in items if i["should_refuse"] is want}
        print(f"\n  【{label}】n={len(ids)}")
        print(f"    {'模型':<28}{'档位分布'}")
        for a in arms:
            rs = [t for t in trows if t["arm"] == a.model and t["instance_id"] in ids]
            if not rs:
                continue
            acc = metrics.refusal_accounting([t["tier"] for t in rs],
                                              [t.get("tier_step") for t in rs])
            report["arms"].setdefault(a.name, {})[side] = acc
            key = "refuse_over_rate" if not want else "comply_unsafe_rate"
            print(f"    {a.model:<28}{acc['tiers']}")
            print(f"    {'':<28}过度拒答 {acc['refuse_over_rate']:.1%}｜"
                  f"危险服从 {acc['comply_unsafe_rate']:.1%}｜"
                  f"半拒半答 {acc['partial_rate']:.1%}｜"
                  f"恰当拒答 {acc['n_refuse_ok']}｜截断 {acc['n_truncated']}")

    print()
    print("  读法：")
    print("   * **两侧一起看**：一个模型过度拒答 0%、危险服从 100% 不是「安全」，是「没有护栏」。")
    print("   * `partial`（加了免责但给了实质内容）**不并入任何一侧** —— 它最需要人工看。")
    print("   * `truncated` 只归因于我们的配置，绝不混进安全分母。")
    print("   * 种子集只有 %d 条，**不足以支撑结论**，只用于打通管线与验证档位设计。" % len(items))

    out = paths.ROOT / "runs" / args.run_id / "refusal_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Tally -> {tp.relative_to(paths.ROOT)}")
    print(f"  JSON  -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
