"""标准答案审计（stage 2）：独立核查标准答案本身有没有缺陷。

为什么必须与归因**分开**
    第一版把 `gold_error` 放进主流程的 8 类里，结果 **53/99 = 53.5%** 被判成
    "标准答案是错的" —— 这不可能成立（那等于说题库过半的错题模型其实是对的）。
    抽查三条，judge 犯了三种不同的错：

      1. `prox-zh-446`（线性规划）：judge **自己把题做对了、算出的是标准答案**，
         却说成"模型答案 F 是正确的" —— 模型实际选的是 A。
         **它混淆了「正确答案」与「模型的答案」。**
      2. `prox-zh-148`（法律）：judge 用**自己未必可靠的法律意见**去否定标准答案。
         这是"我不同意"，不是"标准答案有错"。
      3. `prox-zh-316`（否定题）：两个选项都错 ⇒ 应是"该题不可答"，不是 gold_error。

    根因是提示词写偏了：写了「不要因为标准答案标了某一项就默认它一定对」，
    等于**主动鼓励它不信任标准答案**（与 C7 judge v1 同一类错误）。

现在的做法
    * 主流程只判 7 类，**明确告诉它"标准答案是否有缺陷不是你的事"**。
    * 本脚本独立跑一遍，**不看模型作答**，只问"标准答案有没有缺陷"。
    * 缺陷只允许三类，且**必须指名 + 给出可被第三方复核的依据**：
      `D1` 选项重复 / `D2` 多解 / `D3` 事实错误。
    * 明确列出四条"不算缺陷"（我认为别的选项更好 / 解释不够详尽 / 学界有争议 /
      模型推理更充分），并说明**误判的代价远大于漏判**。
    * 对**全部**条目跑，而不是只跑主流程标为可疑的 —— 这样得到的是**无偏基准率**。

用法
    .venv\\Scripts\\python.exe scripts\\run_gold_audit.py
    .venv\\Scripts\\python\\python.exe scripts\\run_gold_audit.py --report-only
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

from meval import build_set, client, execute, paths  # noqa: E402
from meval.types import Config, PromptSpec  # noqa: E402

PROMPT = "prompts/judge_gold_audit.txt"
LETTERS = "ABCDEFGHIJKLMNOP"
DEFECT_RE = re.compile(r"缺陷\s*[:：]\s*[`*]*\s*(无|D1|D2|D3)\b", re.IGNORECASE)

# 我们已用程序扫出的「选项文本重复」题 —— 作为 D1 的**独立对照**（不由 judge 决定）
KNOWN_DUP = {"prox-zh-88", "prox-zh-156", "prox-zh-163", "prox-zh-282", "prox-zh-515"}


def parse_defect(text: str) -> str:
    ms = list(DEFECT_RE.finditer(text or ""))
    return ms[-1].group(1).upper() if ms else "UNCLEAR"


def main() -> int:
    ap = argparse.ArgumentParser(description="标准答案审计")
    ap.add_argument("--run-id", default="c2-full")
    ap.add_argument("--out-run", default="gold-audit")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    review = [json.loads(l) for l in
              (paths.ROOT / "runs" / args.run_id / "badcase_review.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    # 按题去重（同一道题可能被多个模型答错，但题集缺陷只需审一次）
    seen, items = set(), []
    for r in review:
        if r["instance_id"] not in seen and r["instance_id"] in pool:
            seen.add(r["instance_id"])
            items.append(r["instance_id"])

    judge_model = paths.judge_model()
    base = paths.judge_base_url()
    tmpl = (paths.ROOT / PROMPT).read_text(encoding="utf-8")

    print("=" * 78)
    print(f"标准答案审计（stage 2）   out_run={args.out_run}")
    print("=" * 78)
    print(f"  judge 模型 : {judge_model}")
    print(f"  待审题目   : {len(items)}（按题去重；同一题被多个模型答错只审一次）")
    print(f"  已知文本重复题: {sorted(KNOWN_DUP & set(items))}")

    specs = []
    for iid in items:
        inst = pool[iid]
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(inst.options))
        specs.append(PromptSpec(
            run_id=args.out_run, config=Config.C4, arm=judge_model,
            instance_id=iid, rep=0,
            prompt_text=(tmpl.replace("{question}", inst.question)
                             .replace("{options}", opts)
                             .replace("{gold}", "".join(LETTERS[g] for g in inst.gold_indices))),
            params={"max_tokens": 512, "temperature": 0.1, "reasoning_effort": "none"}))

    if not args.report_only:
        cl = client.ChatClient(base_url=base, api_key=paths.judge_api_key())
        stats = execute.run_specs(run_id=args.out_run, specs=specs,
                                  arms_concurrency={judge_model: args.concurrency},
                                  limit=args.limit, client=cl)
        print(f"  执行       : 失败 {stats.get('failed')}")
        execute.write_manifest(
            run_id=args.out_run, configs=[],
            arms=[{"name": judge_model, "params": {"max_tokens": 512,
                                                   "temperature": 0.1,
                                                   "reasoning_effort": "none"}}],
            data_hash=_sha16(paths.data() / "eval_set.jsonl"),
            models_snapshot=[judge_model],
            note=f"标准答案审计（stage 2）｜题集 {len(items)} 题｜judge={judge_model}",
            stats=stats)

    judged = {r.instance_id: (r.content or "")
              for r in execute.load_results(args.out_run, Config.C4) if r.success}
    verdicts = {iid: parse_defect(c) for iid, c in judged.items()}

    print()
    print("=" * 78)
    print("审计结果")
    print("=" * 78)
    n = len(verdicts)
    ok = sum(1 for v in verdicts.values() if v == "无")
    print(f"  已审 {n} 题｜判「无缺陷」{ok}（{ok/n:.1%}）｜UNABLE {sum(1 for v in verdicts.values() if v=='UNCLEAR')}")
    for k, v in Counter(verdicts.values()).most_common():
        if k != "无":
            print(f"    {k:<8}{v:>4}")

    flagged = [iid for iid, v in verdicts.items() if v.startswith("D")]
    print(f"\n  判为有缺陷的 {len(flagged)} 题：")
    for iid in flagged:
        inst = pool[iid]
        tag = "  ← 程序也扫到文本重复 ✅" if iid in KNOWN_DUP else ""
        print(f"    {iid}  [{verdicts[iid]}]  {inst.question[:52]}{tag}")
        print(f"        {judged[iid].strip().splitlines()[-1][:110] if judged[iid].strip() else ''}")

    miss = KNOWN_DUP & set(verdicts) - set(flagged)
    if miss:
        print(f"\n  ⚠️ 程序扫到文本重复、但 judge 判「无缺陷」的：{sorted(miss)}")
        print(f"     ⇒ 这**不算 judge 错**：文本重复只有在标准答案位于重复组内时才成缺陷。")

    print()
    print("  读法：")
    print("   * 判「D1/D2/D3」的条目**仍需人工确认** —— judge 是提议者，不是裁决者。")
    print("   * 与程序扫出的文本重复题对照，可交叉验证。")
    print("   * 这一遍**不看模型作答**，所以得到的是标准答案缺陷的**无偏基准率**。")

    out = paths.ROOT / "runs" / args.out_run / "gold_audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "judge_model": judge_model, "n_audited": n,
        "distribution": dict(Counter(verdicts.values())),
        "flagged": [{"instance_id": i, "defect": verdicts[i],
                     "reason": judged[i].strip().splitlines()[-1][:200],
                     "program_detected_dup": i in KNOWN_DUP} for i in flagged],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON -> {out.relative_to(paths.ROOT)}")
    return 0


def _sha16(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
