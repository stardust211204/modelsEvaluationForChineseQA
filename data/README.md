# data/ 目录：数据来源与许可

> ⚠️ **`data/` 下的数据不属于本项目代码的授权范围。**
> 仓库根目录的 `LICENSE`（MPL-2.0）**只覆盖本项目自己写的代码与文档**；
> 本目录里的题集与派生件保留其**上游协议**，逐份列在下面。
> 商用、再分发或二次发布前，请按表中的「上游许可」一栏自行确认。

---

## 1 逐份清单

| 文件 | 来源 | 上游许可 |
| --- | --- | --- |
| `eval_set.jsonl`（588 题） | MMLU-ProX 中文切片 —— 代码实际从 `li-lab/MMLU-ProX-Lite`（`lang=zh`、`split=test`）加载 | **MIT**（MMLU-ProX 上游声明） |
| `valid_pool.jsonl` | 同上，`split=valid`，作 few-shot 示例池 | **MIT** |
| `difficulty_proxy_raw.jsonl` | 同上，难度代理打分结果 | **MIT** |
| `e2_relabel_set.jsonl` / `.meta.json` | 由 `eval_set.jsonl` 派生（选项标签替换实验） | 继承 **MIT** |
| `e2b_relabel_set.jsonl` / `.meta.json` | 同上（另一组对照） | 继承 **MIT** |
| `ceval_set.jsonl`（583 题） | C-Eval（`ceval/ceval-exam`） | **CC-BY-NC-SA-4.0** ⚠️ |
| `orbench_hard_raw.csv` / `orbench_toxic_raw.csv` | OR-Bench 原始子集（Hard 1,319 / Toxic 655） | **CC-BY-4.0** |
| `orbench_set.jsonl` / `orbench_pilot.jsonl` | 由上面两个 CSV 整理而来 | 继承 **CC-BY-4.0** |
| `refusal_set.jsonl` / `refusal_raw.json` / `refusal_over_raw.json` / `refusal_under_raw.json` | 同上（安全双向评测的题面与中间件） | 继承 **CC-BY-4.0** |
| `frozen.json` | **本项目生成** —— 题集与模板的冻结哈希 | MPL-2.0 |
| `raw_evidence/*.jsonl`（9 份） | **本项目生成** —— 跑批前探针的原始 API 响应 | MPL-2.0 |

---

## 2 ⚠️ C-Eval 的 NC 与 SA 是硬约束

`ceval_set.jsonl` 来自 **CC-BY-NC-SA-4.0** 的材料，两点必须遵守：

- **NC（NonCommercial，非商业）**：不得用于商业目的。本项目为非商业的求职作品集，
  符合该条；**若你要把它用于商业产品，请自行取得授权**。
- **SA（ShareAlike，相同方式共享）**：对这份数据的**再分发必须以相同协议进行** ——
  也就是说，你不能再以 MPL 或 MIT 的名义发布它。

**这也解释了为什么仓库根目录的 LICENSE 不能覆盖 `data/`**：
MPL 是软件协议，无法用于 CC 授权的数据；一份 LICENSE 盖全仓库会与上游协议冲突。

---

## 3 出处

| 题集 | 仓库 | 论文 |
| --- | --- | --- |
| MMLU-ProX | `li-lab/MMLU-ProX-Lite`（HuggingFace） | arXiv:2503.10497（EMNLP 2025） |
| C-Eval | `ceval/ceval-exam`（HuggingFace） | arXiv:2305.08322（NeurIPS 2023） |
| OR-Bench | `bench-llms/or-bench`（HuggingFace） | arXiv:2405.20947（ICML 2025） |

> **关于 MMLU-ProX 与 MMLU-ProX-Lite**：本项目加载的是 HuggingFace 上的
> `li-lab/MMLU-ProX-Lite`，它是 MMLU-ProX 的中文切片 ——
> 内容出自 MMLU-ProX，题数与全量集不同。
> MMLU-ProX 主仓已在 HuggingFace 上声明 **MIT**；`-Lite` 这个发布仓库本身
> 未单独标注协议，故此处按上游 MMLU-ProX 的 MIT 归口，并保留出处链接以便核对。

---

## 4 派生件与原始响应分别是什么

- **派生件**（`e2*`、`orbench_*`、`refusal_*`）：只做**格式整理与抽样**，未改动题目内容；
  许可跟随其上游。
- **本项目生成**（`frozen.json`、`raw_evidence/`）：不含第三方题面以外的内容。
  `raw_evidence/` 里是**各家模型的 API 原始返回**，由本项目调用产生；
  模型输出的一般性再分发限制请参考各厂商服务条款。

---

*若发现上表的许可标注与上游最新声明不符，以上游为准，并欢迎提 issue 更正。*
