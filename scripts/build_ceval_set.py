"""把 C-Eval 抽成与 MMLU-ProX **同构**的 Instance[]，用于**跨题库交叉验证**。

为什么加 C-Eval（而不是换掉 MMLU-ProX）
    MMLU-ProX-Lite 最大的短板是**它是翻译版**（实测 5/588 题的选项在中文里变成完全相同，
    英文原版 0 处）。而 C-Eval 是**中文原生**：

      > 「the **first comprehensive Chinese evaluation suite** …
      >   to assess advanced knowledge and reasoning abilities … in a **Chinese context**」

    两者角色不同，**互补而非替代**：
      * MMLU-ProX：**10 选项** ⇒ 保留了「选项数」「位置敏感性」「选项混淆」这些结构实验
      * C-Eval：**4 选项 + 52 学科 + 中文原生** ⇒ 检验「排名结论是否只在某一个题库上成立」

    ⇒ **同一个模型排名如果在两个独立题库上一致，那个结论就硬得多。**
      这比"换一个更权威的题库"更有价值 —— 因为换题库只是换一批未知缺陷。

★ 关于 split 的选择
    C-Eval 官方站的 test 标签按设计不公开（需提交榜单）。而 HF `ceval/ceval-exam`
    （官方命名空间，44,206 下载 / 316 点赞 / CC-BY-NC-SA-4.0 / 与官方仓库同日更新）
    **三个 split 都带 `answer` 字段**。本脚本默认用 `test`（12,342 题，每学科约 237 题，
    足以做分学科画像）；`val`（1,346 题）作为可选。

    ⚠️ 公开标签 ⇒ 存在**训练污染**可能。但这一点对所有公开基准都成立，
       而且我们另有反证手段：C6 的选项乱序实验 —— 若模型是"背答案"，
       换顺序不该改变它的选择（实测答案改变 28–53%，指向重新推理而非查表）。

用法
    .venv\\Scripts\\python.exe scripts\\build_ceval_set.py                 # 抽 588 题
    .venv\\Scripts\\python.exe scripts\\build_ceval_set.py --n 588 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import datasets  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

datasets.disable_progress_bars()
datasets.utils.logging.set_verbosity_error()

from meval import build_set, paths  # noqa: E402
from meval.types import Instance, Source, Split  # noqa: E402

REPO = "ceval/ceval-exam"
LETTERS = "ABCD"


def subjects() -> list[str]:
    info = HfApi().dataset_info(REPO)
    return [c["config_name"] for c in (info.cardData or {}).get("configs", [])]


def load_all(split: str) -> list[Instance]:
    """把 52 个学科全部读进来，转成 Instance。

    ★ 逐学科打印进度并容错：第一版没有进度输出，卡住时**无法定位是哪个学科**，
      只能整体杀掉重来。评测脚本涉及网络时，**进度可见性是必需品不是奢侈品**。
    """
    out: list[Instance] = []
    bad = 0
    subs = subjects()
    print(f"  学科数 {len(subs)}，逐个加载（split={split}）…")
    for n, subj in enumerate(subs, 1):
        try:
            ds = datasets.load_dataset(REPO, subj, split=split)
        except Exception as e:
            print(f"    [{n:>2}/{len(subs)}] {subj:<24} ❌ {type(e).__name__}")
            continue
        kept = 0
        for r in ds:
            opts = [(r.get(L) or "").strip() for L in LETTERS]
            if any(not o for o in opts):
                bad += 1
                continue
            ans = (r.get("answer") or "").strip().upper()
            if ans not in LETTERS:
                bad += 1
                continue
            q = (r.get("question") or "").strip()
            if not q:
                bad += 1
                continue
            out.append(Instance(
                id=f"ceval-{subj}-{r.get('id')}",
                question=q,
                options=opts,
                gold_indices=[LETTERS.index(ans)],
                split=Split.TEST,
                # ★ n_options 恒为 "4" —— C-Eval 没有选项数变异，
                #   所以这一维在 C-Eval 上无法做分层（这是它的结构性差异，不是缺陷）
                stratum={"subject": subj, "n_options": "4", "difficulty": "unknown"},
                source=Source.PUBLIC,
                note=f"C-Eval {split}｜{subj}",
            ))
            kept += 1
        print(f"    [{n:>2}/{len(subs)}] {subj:<24} {kept:>5} 题")
    print(f"  跳过脏行 {bad} 条，累计 {len(out)} 题")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 C-Eval 抽样集（跨题库交叉验证用）")
    ap.add_argument("--n", type=int, default=588, help="抽样题数（与 MMLU-ProX 同量级）")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--seed", type=int, default=build_set.SEED)
    ap.add_argument("--out", default="ceval_set.jsonl")
    args = ap.parse_args()

    print("=" * 78)
    print("构建 C-Eval 抽样集")
    print("=" * 78)
    print(f"  来源 : HF {REPO}（split={args.split}）")
    print("  说明 : 中文**原生**基准，与 MMLU-ProX（翻译版）交叉验证模型排名")

    pool = load_all(args.split)
    print(f"  全量 : {len(pool)} 题，{len({i.stratum['subject'] for i in pool})} 个学科")

    # ---- 按学科分层抽样（每学科按比例，至少 1 题）--------------------------
    from collections import defaultdict
    import random

    bysub: dict[str, list[Instance]] = defaultdict(list)
    for i in pool:
        bysub[i.stratum["subject"]].append(i)
    rng = random.Random(args.seed)
    total = len(pool)
    picked: list[Instance] = []
    for subj in sorted(bysub):
        rows = bysub[subj]
        k = max(1, round(args.n * len(rows) / total))
        picked.extend(rng.sample(rows, min(k, len(rows))))
    rng.shuffle(picked)
    picked = picked[:args.n]

    # ---- 抽样保真度自检（样本 vs **本题库完整池子**）------------------------
    # ★ 原先传的是 MMLU-ProX 的 eval_set：两者**学科口径不同**（C-Eval 52 学科
    #   vs MMLU-ProX 14 学科）、**选项数口径也不同**（恒 4 vs 多为 10），
    #   所以那两个 gap 都没有意义 —— 等于这次自检**根本没做**。
    #   而自检的本意（见 comparability_report 文档串）是"证明抽出的样本与
    #   完整池子分布一致"，参照物必须是**本题库的池子**。
    rep = build_set.comparability_report(pool, picked)
    nsub_pool = len({i.stratum["subject"] for i in pool})
    print(f"\n  抽样 : {len(picked)} 题，{len({i.stratum['subject'] for i in picked})}/{nsub_pool} 个学科")
    print(f"  抽样保真度（对 C-Eval 池子 {len(pool)} 题）: "
          f"选项数 max_gap={rep['n_options']['max_abs_gap']}  "
          f"学科 max_gap={rep['subject']['max_abs_gap']}")
    worst = max(rep["n_options"]["max_abs_gap"], rep["subject"]["max_abs_gap"])
    if worst > 0.15:
        print("    ⚠️ 分布偏差 >15pp，分层抽样可能失真，请检查 --n / 种子")
    else:
        print("    ✅ 未超 15pp 阈值")

    h = build_set.freeze(picked, paths.data() / args.out)
    print(f"\n  已冻结: data/{args.out}  ({len(picked)} 题)  hash={h}")
    print("\n  下一步：跑 scripts/run_config.py --dataset ceval_set.jsonl（若尚未支持，需先加参数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
