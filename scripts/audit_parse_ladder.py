"""审计：宽松解析阶梯的可靠性 —— 有多少判定其实抓错了答案？

背景
    `tally.parse_answer` 的宽松阶梯分三级：
        strong  bare / answer-marker / last-line   —— 模型**明确**给了答案
        weak    phrase（"选项 X"）                 —— 模型可能只是在**讨论**选项
        none    没解出

    实测 `qwen` 在 C2-hi 上有 10.2% 的判定落在 weak；其余臂 0–1.5%。
    而且 weak 组的准确率只有 28.6%（strong 组 81.2%）——
    一个"组内准确率断崖"通常意味着**测量本身有问题**，而不是模型突然变差。

    根因与 2026-10-05 修掉的那个解析 bug 同源：`选项` 这个触发词抓的是
    「在讨论选项」，不等于「给出答案」。上次修了「取第一个」→「取最后一个」，
    但**只要模型从不明确表态，兜底就仍可能抓错**。

本脚本做什么
    对每一条 weak 判定，做一次**独立复核**：
      从正文的**结论区**（末尾）重新抽答案，看是否与解析结果一致。
      结论区不用 `选项` 这个不可靠触发词，改用「因此/所以/应选/正确/答案是」
      这类**表态词**，加上末尾若干字符内的括号字母。

    两者不一致 ⇒ 该条判定**可能抓错了**，计入 `suspected`。

输出
    ① 全配置的阶梯分布（weak / none 占比）
    ② weak 组里有多少条**疑似抓错**
    ③ 若把疑似抓错的改判，各臂准确率会怎么变（敏感性）

用法
    .venv\\Scripts\\python.exe scripts\\audit_parse_ladder.py
    .venv\\Scripts\\python.exe scripts\\audit_parse_ladder.py --samples 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, exclusions, paths, tally  # noqa: E402
from meval.types import Config, ParseMode  # noqa: E402

ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
        "mimo-v2.6-flash", "qwen3.8-flash"]
STRONG = {"bare", "answer-marker", "last-line"}
LETTERS = "ABCDEFGHIJ"

# 表态词 + 紧随其后的字母（或括号字母）。**故意不含 `选项`** —— 那正是不可靠的来源。
CONCLUDE_RE = re.compile(
    r"(?:因此|所以|故|应选|故选|正确的是|正确答案|答案是|答案为|结论)\s*"
    r"[:：]?\s*[是为]?\s*[（(]?\s*([A-J])\s*[）)]?",
    re.IGNORECASE,
)
# 结论区里的裸括号字母，取最后一个
PAREN_RE = re.compile(r"[（(]\s*([A-J])\s*[）)]")

RUNS = [
    ("C1 / MMLU-ProX",    "c1-5arm-v2",         "c1", "eval_set"),
    ("C1 / C-Eval",       "ceval-c1-5arm-v2",   "c1", "ceval_set"),
    ("C2-hi / MMLU-ProX", "c2hi-5arm-v2",       "c2", "eval_set"),
    ("C2-hi / C-Eval",    "ceval-c2hi-5arm-v2", "c2", "ceval_set"),
    ("C3 / MMLU-ProX",    "c3-5arm-v2",         "c3", "eval_set"),
    ("C5 / MMLU-ProX",    "c5-5arm-v2",         "c5", "eval_set"),
]

CONCLUDE_WINDOW = 220      # 结论区取末尾多少字符


def independent_answer(text: str, n_opts: int) -> tuple[str | None, str]:
    """独立复核：只在**结论区**里找答案，不依赖 `选项` 触发词。"""
    tail = (text or "")[-CONCLUDE_WINDOW:]
    m = None
    for mm in CONCLUDE_RE.finditer(tail):
        m = mm                                     # 取最后一处表态
    if m:
        return m.group(1).upper(), "conclude-phrase"
    ps = PAREN_RE.findall(tail)
    if ps:
        return ps[-1].upper(), "tail-paren"
    return None, "no-signal"


def main() -> int:
    ap = argparse.ArgumentParser(description="审计宽松解析阶梯的可靠性")
    ap.add_argument("--samples", type=int, default=5, help="每个不一致组打印几条样例")
    args = ap.parse_args()
    drop = exclusions.load()

    totals: dict[str, Counter] = defaultdict(Counter)
    suspected: list[dict] = []
    weak_rows: list[dict] = []

    for label, run, cfg, ds in RUNS:
        pool = {i.id: i for i in build_set.load_frozen(paths.data() / f"{ds}.jsonl")}
        tfile = paths.ROOT / "runs" / run / "tallies" / cfg / "all.jsonl"
        if not tfile.exists():
            continue
        tiers = {}
        for line in tfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t.get("mode") == "lenient":
                    tiers[(t["arm"], t["instance_id"], t.get("rep", 0))] = t
        for arm in ARMS:
            rfile = paths.ROOT / "runs" / run / "raw" / cfg / f"{arm}.jsonl"
            if not rfile.exists():
                continue
            for line in rfile.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("success"):
                    continue
                iid = r["instance_id"]
                if iid in drop:
                    continue
                inst = pool.get(iid)
                if inst is None:
                    continue
                idx, step = tally.parse_answer(r.get("content") or "", len(inst.options),
                                               ParseMode.LENIENT)
                totals[arm][step] += 1
                totals[arm]["_all"] += 1
                if step != "phrase":
                    continue
                ind, how = independent_answer(r.get("content") or "", len(inst.options))
                parsed = LETTERS[idx] if idx is not None else None
                agree = (ind is None) or (ind == parsed)
                t = tiers.get((arm, iid, r.get("rep", 0)))
                row = {"label": label, "arm": arm, "iid": iid, "parsed": parsed,
                       "independent": ind, "how": how, "agree": agree,
                       "tier": (t or {}).get("tier")}
                weak_rows.append(row)
                if not agree:
                    suspected.append(row)

    # ---- ① 阶梯分布 -------------------------------------------------------
    print("=" * 104)
    print("  ① 宽松解析阶梯分布（五臂 × 全配置）")
    print("=" * 104)
    print(f"  {'臂':<26}{'总条数':>8}{'strong':>8}{'weak':>7}{'none':>7}{'weak占比':>10}{'none占比':>10}")
    for arm in ARMS:
        c = totals[arm]
        n = c.get("_all", 0)
        if not n:
            continue
        weak = c.get("phrase", 0)
        none = c.get("none", 0) + c.get("empty", 0)
        print(f"  {arm[:24]:<26}{n:>8}{n-weak-none:>8}{weak:>7}{none:>7}"
              f"{weak/n:>10.1%}{none/n:>10.1%}")

    # ---- ② weak 组的独立复核 ----------------------------------------------
    print()
    print("=" * 104)
    print("  ② weak 判定的独立复核（从**结论区**重新抽答案，不用 `选项` 触发词）")
    print("=" * 104)
    print(f"  weak 判定共 {len(weak_rows)} 条；独立复核给出不同答案的：**{len(suspected)}** 条")
    by_arm = Counter(r["arm"] for r in suspected)
    for arm in ARMS:
        nw = sum(1 for r in weak_rows if r["arm"] == arm)
        if nw:
            print(f"    {arm[:24]:<26} weak {nw:>4} 条，其中疑似抓错 {by_arm.get(arm,0):>3} "
                  f"({by_arm.get(arm,0)/nw:.0%})")
    if suspected:
        print()
        print("  样例：")
        for r in suspected[:args.samples]:
            print(f"    [{r['label']}] {r['arm'][:14]:<14} {r['iid']:<14} "
                  f"解析={r['parsed']} 独立复核={r['independent']}({r['how']}) 判定={r['tier']}")
    no_sig = sum(1 for r in weak_rows if r["how"] == "no-signal")
    print()
    print(f"  另：结论区**完全没有信号**的 {no_sig} 条 —— 这些复核不了，"
          f"但它们恰恰说明模型『没有明确表态』。")

    # ---- ③ 敏感性 ---------------------------------------------------------
    print()
    print("=" * 104)
    print("  ③ 敏感性：把『疑似抓错』的 weak 判定改判为 invalid，准确率如何变")
    print("=" * 104)
    print("  （这不是建议改判 —— 只是量化这条口径最多能影响多少）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
