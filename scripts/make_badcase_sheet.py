"""把 badcase 复核清单转成**人能读的工作单**。

为什么需要这一步
    `runs/{run}/badcase_review.jsonl` 有 104 条，光模型推理就 **约 7.9 万字**
    （中位 677 字/条，最长 2151 字）。原始 JSONL 既没法读也没法标。
    人工复核的真正瓶颈不是判断难度，是**阅读与记录的摩擦**。

产物（两个，配套使用）
    badcase_review.md            编号 R001–R104，含题面/选项/标准答案/模型作答/完整推理
    badcase_labels_template.csv  只有 id 与空格子，填完交回来即可合并

用法
    .venv\\Scripts\\python.exe scripts\\make_badcase_sheet.py
    .venv\\Scripts\\python.exe scripts\\make_badcase_sheet.py --run-id c2-full --config c2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import build_set, execute, paths  # noqa: E402
from meval.types import Config  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOP"

LABEL_HELP = """\
| 标签 | 含义 | 判据 |
| --- | --- | --- |
| `knowledge_gap` | 知识缺失 | 推理里**没出现**任何与正确项相关的概念 |
| `reasoning_error` | 推理链错误 | 提到了正确概念，但推导过程有明确错误 |
| `option_confusion` | 选项混淆 | 理解正确，选中的是与正确项**相似或相邻**的干扰项 |
| `question_misread` | 误读题干 | 在回答**另一个问题** |
| `detail_omission` | 遗漏关键细节 | 漏掉题干里的限定条件（"除了"/"最不"/"在 X 条件下"） |
| **`gold_error`** | **金标准有误** | 模型的答案**其实是对的**，是标准答案错了 ★ 最值得留意 |
| `format_parse` | 格式/解析 | 答案实质正确，但被我们的口径判成错 |
| `cannot_tell` | 无法判断 | 信息不足 |

**填写要点**
* 只能填上表里的 8 个英文标签之一，不要自创。
* `gold_error` 请格外留意 —— 它一旦成立，**已发布的所有准确率数字都要重算**。
  判断方法：把模型的选择当成对的，看它能不能自圆其说；若能，标 `gold_error` 并在备注写明理由。
* `cannot_tell` 是合法答案，不要为了凑标签硬选。
* 备注写**一句话**即可，不用长。
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 badcase 人工复核工作单")
    ap.add_argument("--run-id", default="c2-full")
    ap.add_argument("--config", default="c2")
    args = ap.parse_args()

    cfg = Config(args.config)
    rows = [json.loads(l) for l in
            (paths.ROOT / "runs" / args.run_id / "badcase_review.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    res = {(r.arm, r.instance_id): r for r in execute.load_results(args.run_id, cfg)}

    md: list[str] = []
    md.append(f"# badcase 人工复核工作单（{len(rows)} 条）\n")
    md.append(f"> 来源：`runs/{args.run_id}/badcase_review.jsonl`｜配置 `{args.config}`（自由作答口径）\n")
    md.append("> **怎么用**：逐条读，把判定填进同目录的 "
              "`badcase_labels_template.csv`（只有 id 与空格子）。填完交回即可合并。\n")
    md.append("\n## 标签集与判据\n")
    md.append(LABEL_HELP)
    md.append("\n## 全局提示\n")
    md.append(f"* 分臂：{dict(Counter(r['arm'] for r in rows))}")
    md.append(f"* 学科：{dict(Counter(r['subject'] for r in rows))}")
    md.append("* `启发式` 那一行**仅供参考、且已实测不可靠**（它实际只在区分"
              "回复长短）—— 请独立判断，不要被它带偏。\n")
    md.append("---\n")

    csv_rows = []
    for n, r in enumerate(rows, 1):
        rid = f"R{n:03d}"
        iid, arm = r["instance_id"], r["arm"]
        inst = pool.get(iid)
        m = res.get((arm, iid))
        md.append(f"\n### {rid} · `{arm}` · {r['subject']}\n")
        if inst is None:
            md.append(f"_(题集里找不到 {iid})_\n")
            continue
        md.append(f"**题**：{inst.question}\n")
        for i, o in enumerate(inst.options):
            mark = " ✅" if i in inst.gold_indices else ""
            md.append(f"- {LETTERS[i]}. {o}{mark}")
        gold = "".join(LETTERS[g] for g in inst.gold_indices)
        pick = LETTERS[r["model_answer_index"]] if r.get("model_answer_index") is not None else "（判不出）"
        md.append(f"\n**标准答案**：{gold}　|　**模型选了**：**{pick}**")
        md.append(f"\n<sub>启发式={r.get('heuristic_attribution')}｜"
                  f"提到正确项={r.get('mentions_gold_text')}｜相邻={r.get('adjacent_to_gold')}｜"
                  f"解析阶梯={r.get('tier_step')}</sub>\n")
        md.append("\n**模型作答全文**：\n")
        body = (m.content if m else "") or ""
        md.append("```text")
        md.append(body.replace("```", "'''"))
        md.append("```")
        md.append(f"\n**✍️ {rid} 判定**：______　**备注**：______\n")
        md.append("\n---")
        csv_rows.append({"id": rid, "instance_id": iid, "arm": arm,
                         "confirmed_attribution": "", "reviewer_note": ""})

    outdir = paths.ROOT / "runs" / args.run_id
    mdp = outdir / "badcase_review.md"
    mdp.write_text("\n".join(md) + "\n", encoding="utf-8")

    csvp = outdir / "badcase_labels_template.csv"
    with csvp.open("w", encoding="utf-8-sig", newline="") as f:   # BOM 让 Excel 正确识别中文
        w = csv.DictWriter(f, fieldnames=["id", "instance_id", "arm",
                                          "confirmed_attribution", "reviewer_note"])
        w.writeheader()
        w.writerows(csv_rows)

    total = sum(len((res.get((r["arm"], r["instance_id"])).content or "")
                    if res.get((r["arm"], r["instance_id"])) else "") for r in rows)
    print("=" * 78)
    print("badcase 人工复核工作单")
    print("=" * 78)
    print(f"  条数     : {len(rows)}")
    print(f"  阅读量   : {total:,} 字（中位 {sorted(r['content_len'] for r in rows)[len(rows)//2]} 字/条）")
    print(f"  分臂     : {dict(Counter(r['arm'] for r in rows))}")
    print()
    print(f"  工作单   -> {mdp.relative_to(paths.ROOT)}   （读这个）")
    print(f"  填写表   -> {csvp.relative_to(paths.ROOT)}   （填这个，Excel 可直接打开）")
    print()
    print("  回填后跑：")
    print(f"    .venv\\Scripts\\python.exe scripts\\merge_badcase_labels.py "
          f"--run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
