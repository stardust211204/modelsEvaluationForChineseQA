"""留一法共识难度：纯后处理，零 API 调用。

为什么不能直接用「多模型共识」定难度
    用三个模型的作答定义难度，再按该难度报**同一批模型**的准确率，是循环论证：
    "三个模型都错的题"必然在三个模型上都低分 —— 那是个恒等式，不是发现。
    更糟的是它会**看起来很漂亮**（梯度极陡），因为梯度是构造出来的。

留一法（leave-one-out）
    报某模型的难度分层时，难度**只由另外两个模型**的作答定义。
    目标模型的答案没有参与给自己贴标签，循环性被切断。

它仍然不是完全独立的难度度量
    三个模型在同一批题上的失败是相关的，所以留一法只**减弱**、不消除循环性。
    报告里必须这样声明。真正独立的代理是 `n_options` ——
    选项数由题面决定，与任何模型的作答无关。

本脚本同时打印两种口径，好让循环性**被看见**而不是被争论。

用法
    .venv\\Scripts\\python.exe scripts\\difficulty_loo.py
    .venv\\Scripts\\python.exe scripts\\difficulty_loo.py --run-id c1-full --write

产物
    runs/{run_id}/difficulty_loo.json     各模型的留一法难度标签 + 分层准确率
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import metrics, paths  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402

CORRECT, WRONG = "correct", "wrong"
ORDER = ("easy", "mid", "hard")


def consensus_label(tiers: list[str]) -> str | None:
    """把若干模型在同一题上的判级压成一个难度标签。

    全部判对 = easy；全部判错 = hard；有对有错 = mid。
    只要有一个模型没给出二元判定（invalid / truncated），该题标 unknown ——
    "没人答上来"和"很难"是两件事，不能混。
    """
    if not tiers:
        return None
    b = [t for t in tiers if t in (CORRECT, WRONG)]
    if len(b) != len(tiers):
        return None
    n_ok = sum(1 for t in b if t == CORRECT)
    if n_ok == len(b):
        return "easy"
    if n_ok == 0:
        return "hard"
    return "mid"


def main() -> int:
    ap = argparse.ArgumentParser(description="留一法共识难度")
    ap.add_argument("--run-id", default="c1-full")
    ap.add_argument("--config", default="c1")
    ap.add_argument("--write", action="store_true", help="落盘 JSON")
    args = ap.parse_args()

    cfg = Config(args.config)
    rows = [t for t in metrics.load_tallies(args.run_id, cfg)
            if t["mode"] == ParseMode.STRICT.value]
    if not rows:
        print(f"!! runs/{args.run_id}/tallies/{cfg.value}/all.jsonl 为空")
        return 2

    arms = sorted({t["arm"] for t in rows})
    insts = sorted({t["instance_id"] for t in rows})
    tier: dict[tuple[str, str], str] = {(t["arm"], t["instance_id"]): t["tier"] for t in rows}

    print("=" * 78)
    print(f"留一法共识难度   run_id={args.run_id}   配置={cfg.value}（严格口径）")
    print("=" * 78)
    print(f"  模型 {len(arms)} 个 ｜ 题 {len(insts)} 道\n")

    def acc_of(target: str, labels: dict[str, str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for lab in ORDER:
            qs = [q for q, l in labels.items() if l == lab]
            ok = sum(1 for q in qs if tier.get((target, q)) == CORRECT)
            bad = sum(1 for q in qs if tier.get((target, q)) == WRONG)
            out[lab] = {"n": ok + bad, "correct": ok,
                        "acc": round(ok / (ok + bad), 4) if ok + bad else None}
        return out

    detail: dict = {"run_id": args.run_id, "config": cfg.value,
                    "method": "leave-one-out consensus（严格口径）", "arms": {}}

    print("-" * 78)
    print("★ 留一法：给某模型贴难度标签时，**不使用该模型自己的作答**（切断循环）")
    print("-" * 78)
    hdr = "".join(f"{lbl:>14}" for lbl in ORDER)
    print(f"  {'模型':<28}{hdr}")
    for target in arms:
        others = [a for a in arms if a != target]
        labels = {}
        for q in insts:
            lab = consensus_label([tier.get((a, q), "") for a in others])
            if lab:
                labels[q] = lab
        per = acc_of(target, labels)
        cells = "".join(
            f"{(f'{per[l]['acc']:.1%}' if per[l]['acc'] is not None else '—'):>8}"
            f"{f'(n={per[l]['n']})':>6}" for l in ORDER
        )
        print(f"  {target:<28}{cells}")
        detail["arms"][target] = {
            "n_labelled": len(labels),
            "by_difficulty_loo": per,
            "labels": labels,
        }

    print()
    print("-" * 78)
    print("对照 · 全量共识（**含**目标模型自己 —— 循环论证，仅供对照，不得引用）")
    print("-" * 78)
    print(f"  {'模型':<28}{hdr}")
    for target in arms:
        labels = {}
        for q in insts:
            lab = consensus_label([tier.get((a, q), "") for a in arms])
            if lab:
                labels[q] = lab
        per = acc_of(target, labels)
        cells = "".join(
            f"{(f'{per[l]['acc']:.1%}' if per[l]['acc'] is not None else '—'):>8}"
            f"{f'(n={per[l]['n']})':>6}" for l in ORDER
        )
        print(f"  {target:<28}{cells}")
        detail["arms"][target]["by_difficulty_circular"] = per

    print()
    print("读法")
    print("  * 留一法那一组才是可以写进报告的；「全量共识」那一组是**构造出来的梯度**，")
    print("    它越陡，越说明循环论证会把难度效应放大。")
    print("  * 三个模型在同一批题上的失败是相关的，所以留一法只**减弱**、不消除循环性。")
    print("    真正独立的难度代理是 `n_options`（题面决定，与模型无关）。")
    print("  * 原先的本地模型难度代理已失败（见 data/difficulty_proxy_raw.jsonl），")
    print("    这是它的**替代方案**，不是同一件事。")

    if args.write:
        out = paths.ROOT / "runs" / args.run_id / "difficulty_loo.json"
        out.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
