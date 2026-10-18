# modelsEvaluationForChineseQA

**大模型中文问答能力的评测工程实践** —— 以 HELM 的「场景 × 适配 × 指标」三元组为口径基准，自建轻量级双管线评测框架，完整覆盖「评测方案设计 → 评测集构建与清洗 → 双口径判分体系 → 独立裁判抽检复核 → 错题归因与金标审计 → 可复算评测报告」六个工程环节。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)

> **核心文档直达**：[对外评测报告](reports/report_external_helm_polished.md) · [数据逐格溯源清单](reports/report_data_provenance.md) · [全库 85 批次审计台账](reports/RUN_AUDIT.md) · [闭环成果物映射](reports/PROJECT_SUMMARY.md) · [复现与门禁指南](docs/reproduce.md)

---

## 一、评测规模

| 项 | 内容与规模 |
|---|---|
| **被测模型** | 5 个（同厂跨尺寸 + 四厂同档位对照）：`deepseek-v4.1-flash`、`deepseek-v4-pro`、`longcat-2.5-preview-free`、`mimo-v2.6-flash`、`qwen3.8-flash` |
| **评测题集** | MMLU-ProX（588 题 → 剔除 42 道源数据缺陷题，有效计分 **546** 题）、C-Eval（分层抽样 **583** 题）、OR-Bench（全量 **1,974** 题 = Hard 1,319 + Toxic 655） |
| **评测任务** | 9 项：直接作答 / 自由作答 / 作答格式递进 / 5-shot / 重复采样 / 选项乱序 / 选项标签替换 / 思考开关对照 / Judge 抽样复核 |
| **调用总量** | **53,379 次**（成功 53,291，失败 88），对应 **37 个权威结论批次** |
| **计分口径** | 准确率 = `correct ÷ (correct + wrong)`；`invalid`（解析失败）与 `truncated`（生成截断）不计入分母、单独统计报告 |

上表仅统计最终进入报告结论的 **37 个权威批次（53,379 次调用）**。在全周期研发与排障过程中，全库累计执行 **85 个批次、77,512 次模型调用**；其余 48 个批次（逾 2.4 万次调用）涵盖前置参数探针、口径校准，以及因检出方法学缺陷而主动隔离归档的历史批次。全量批次的状态流转与数据可用性核查详见 [reports/RUN\_AUDIT.md](reports/RUN_AUDIT.md) 与 [reports/RUN\_INVENTORY.md](reports/RUN_INVENTORY.md)。

> 各维度的实测结果、单因子受控实验、统计检验与**分场景适配建议**详见 [reports/report\_external\_helm\_polished.md](reports/report_external_helm_polished.md)（§1.3 实测速览 → 第 3 章基线表现 → 第 4 章评测条件敏感度 → §5.1 方法学结论 → §5.3 分场景建议）。

---

## 二、系统架构与对 HELM 的工程修正

### 与 HELM 的关系

斯坦福 HELM 框架已于 2026-06-01 转入维护模式（Maintenance Mode），且其核心假设（开放模型权重 + 基于 `logprob` 的选项概率打分）在商业黑盒 API 与推理模型环境下不再成立。

- **本项目沿用**：分层架构、严格数据契约、截断与异常单列、数据污染显式声明。
- **本项目裁去**：`logprob` 打分路径、16 场景重型套件、多客户端抽象层与前端展示层。

**对 HELM 判分契约的三处工程修正**（详见 [docs/design.md](docs/design.md)）：

| 维度 | HELM 原生处理 | 本项目工程修正 | 修正理由（保障测试归因准确性） |
|---|---|---|---|
| **解析失败** | 并入错误，直接计为答错 | **独立 `invalid` 档**，不计入准确率分母 | 避免把正则覆盖不足或指令格式冲突等「判分器 / 协议问题」误判为模型知识缺陷 |
| **生成截断** | 单独记录，**但仍计入准确率分母** | **独立 `truncated` 档**，不计入分母 | 推理模型将输出预算耗尽在思维链上是常态；将「输出预算设小了」记成「模型答错」会造成系统性归因失真 |
| **多正确答案** | 原生支持 | **同样支持** `gold_indices: list[int]` | 兼容否定式提问或存在争议的多解题目，保证底层数据结构不耦合单选假设 |

### 管线分层与职责边界

![准确率评测管线的分层结构](figures/40_pipeline_main.png)

<details>
<summary>查看五层管线模块清单（<code>src/meval/</code>）</summary>

```
src/meval/
  types.py       核心数据契约（Item / PromptSpec / ModelResult / Tally）
  build_set.py   ① 造题层：分层抽样 + 随机种子固化 + 冻结哈希
  adapt.py       ② 适配层：纯函数 Prompt 渲染 + 哨兵插槽 + 模板指纹计算
  execute.py     ③ 执行层：异步并发 / 指数退避 / 断点续跑 / 原始响应落盘
  tally.py       ④ 判分层：严格·宽松双口径正则阶梯 + 四档判级
  metrics.py     ⑤ 聚合层：四档分账 / McNemar 精确检验 / Cohen κ / 留一法难度
  client.py      OpenAI 兼容客户端（标准参数与厂商私有 extra_body 分流）
  exclusions.py  题集源数据缺陷剔除清单（42 题，带指纹校验）
```

</details>

### 各层职责划分与架构契约

管线按单向依赖严格分层。以下五条工程约束保证实验结果可复现、可审计：

1. **`execute` 层只负责调用与落盘**：仅处理网络并发、退避重试与原始响应持久化，不含任何解析或判分逻辑——确保判分规则无论如何演进都不污染原始采集数据。
2. **`tally` 层为纯离线确定性计算**：仅消费已落盘的 `raw` 原始记录，绝不发起任何外部模型调用——支持对同一批原始响应零成本反复重算。
3. **Few-shot 示例严格取自 `valid` split**：物理隔离 `test` split，从源头杜绝测试集数据泄漏。
4. **准确率全库仅允许唯一分母实现**：分母恒为 `correct + wrong`，结算逻辑唯一收口于 `metrics.tier_accounting()`，跑批日志与绘图脚本均不得另行计算。
5. **高截断场景强制启用统一共同集**：当截断率不可忽略时，强制收敛至所有被测模型共同完成的题目交集，并在所有模型上同步剔除同一批未完成题目。

> **为何严防分母漂移（第 4 条）**：早期跑批日志曾自行按 `correct` ÷ 总题数 打印粗略准确率，一旦下游日志或绘图脚本各自结算，同一批数据就会流出两套看似「都有依据」的准确率——**两套分母共存比单一计算错误更隐蔽，也更难排查**。

### 核心工程决策：`raw` 与 `tally` 物理分离

评测迭代中最常发生返工的是**判分与解析逻辑**（正则阶梯微调、档位边界修正）。将不可变的模型原始输出（`raw`）与可重算的判分结果（`tally`）在物理存储上完全解耦后，修改判分规则**无需重跑任何一次 API**——本项目周期内判分口径迭代多次，API 重跑次数为 **0**，回归验证成本从小时级压缩至秒级。

<details>
<summary>查看运行产物目录布局与分工（<code>runs/</code>）</summary>

```
runs/{run_id}/manifest.json                 # 参数生效对账表、代码版本、数据与模板哈希
runs/{run_id}/raw/{config}/{arm}.jsonl      # 不可变原始层：PromptSpec + ModelResult（大文件走 GitHub Release）
runs/{run_id}/tallies/{config}/{arm}.jsonl  # 可重算判分层：Tally（已随 Git 入库，报告全部指标直接由它生成）
```

</details>

---

## 三、项目中的缺陷与经验

以下六条是跑批与分析过程中实际踩到的典型陷阱，以及最终固化进代码与门禁的工程修法。它们大多属于**黑盒 API 与推理模型评测中极易静默发生**的隐蔽失真：

| # | 评测缺陷与陷阱 | 现象与实测代价 | 工程应对与修法 | 固化防线 |
|---|---|---|---|---|
| **1** | **推理模型空响应静默失败与截断幸存者偏差** | 思维链耗尽输出预算时返回空串，**但 HTTP 状态码仍为 200**；且被截断题系统性偏难（正确率比未截断题低 **28 个百分点**），若直接剔除会使准确率显著虚高 | 按 `finish_reason` 将截断独立归入 `truncated` 档；跨模型对比时强制收敛至各模型共同完成的统一题集 | `tests/test_truncation_bias.py`<br>`tests/test_tally.py` |
| **2** | **裁判模型（Judge）思考未关导致选择性漏判** | 裁判模型隐藏思维链吃光输出预算，导致约 **12%** 的判定拿不到结果，且失败高度集中在规则判错的争议题上——直接统计会人为高估人机一致率 | 判分前显式冻结 Judge 的 `reasoning_effort="none"`，并记入运行元数据 `manifest.json` | `scripts/_check_judge_verdicts.py` |
| **3** | **Few-shot 示例格式压倒正文指令 & 跨集选项数错配** | 首轮示例答案行写成「答案：X」，模型模仿该前缀导致 **87%** 的输出无法通过严格解析，伪造出虚假的格式顺从「增益」；另有 C-Eval 4 选项题误用 10 选项示例导致合法选项范围冲突 | 示例强制仅示范目标输出格式本身（裸字母），且选项数量与目标题严格对齐；跑批前校验模板指纹，不一致即拒绝启动 | `tests/test_prompt_letters.py`<br>`src/meval/adapt.py` |
| **4** | **可判分母不一致时直接相减导致归因反转** | 放宽格式后各档可判条目数相差一倍以上（**276 对 635**），若直接拿各组粗准确率相减，会得出因果方向完全相反的结论 | 跨配置比较一律限制在双方均可判的**配对交集**上，统一执行配对 McNemar 精确检验 | `tests/test_paired.py`<br>`scripts/exp_e6_paired.py` |
| **5** | **跨观测次数翻转率不可直接相除 & 乱序未还原坐标系** | 「是否发生过翻转」随观测次数增加而单调上升，4 次重排与 10 次重复采样不能直接相除；选项乱序若漏还原原题坐标系，会算出虚假的高翻转率 | 改用与观测次数无关的「逐对不一致率」计算比值；乱序后强制映射回原题选项坐标系再做比对 | `tests/test_position_refusal.py`<br>`scripts/_ratio_pairwise.py` |
| **6** | **固定温度 ≠ 输出可复现 & 厂商私有参数冲突** | 探针实测部分厂商 API 直接忽略温度参数；且即便参数固定，同题重复 10 次的全同率最低仍仅有 **28%**；此外各厂商关闭思考的参数名互斥 | 放弃全局统一参数假设，接入新模型前先跑探针并将生效参数冻结至 `configs/arms.yaml`；私有参数统一走 `extra_body` | `tests/test_client_params.py`<br>[docs/param\_matrix.md](docs/param_matrix.md) |

---

## 四、自动化校验

按测试生命周期由前向后排列，上述排障经验均已沉淀为管线中的四层自动化校验机制。

### ① 跑批前探针（Pre-flight Probing）

| 脚本 | 探测内容与防错目标 |
|---|---|
| `scripts/smoke_arms.py` | 各模型端到端连通性与最小可用性冒烟检查 |
| `scripts/probe_params.py` | 实测各模型的**参数生效边界**（温度是否生效、思考模式如何关闭、`max_tokens` 下限） |
| `scripts/probe_distinct.py` | **伪别名行为指纹** —— 针对 `resp_model` 仅回显请求名的问题，用 30 题 × 各模型的答案一致率矩阵反查底层是否为同一模型 |
| `scripts/compare_endpoints.py` | 同一模型在不同 API 端点下的输出一致性核查 |
| `scripts/probe_alias.py` / `probe_new_arm.py` | 新增模型接入前的定点兼容性探测 |

探测结果回填至 [docs/param\_matrix.md](docs/param_matrix.md) 并**冻结**进 `configs/arms.yaml`——不做全局统一参数假设，因为实测证明参数行为跨厂商不可约。

### ② 数据清洗与指纹冻结（Data Hygiene & Fingerprinting）

| 机制 | 说明 |
|---|---|
| `configs/exclusions.json` + `src/meval/exclusions.py` | **42 道源数据缺陷题隔离**（37 题 LaTeX 公式损坏 + 5 题翻译后选项重复）：保持原始题集文件不变，在统计聚合时通过带指纹校验的剔除清单统一滤除（扫描判据见 `scripts/_check_broken_latex.py`，生成脚本见 `scripts/build_exclusions.py`） |
| `data/frozen.json` | 题集抽样与提示词模板的冻结哈希基线；可通过 `scripts/_replay_fingerprints.py` 离线重放校验 |
| `src/meval/adapt.py::prompt_fingerprint()` | **模板指纹门禁**：`scripts/run_config.py` 跑批前强制比对模板指纹，不一致即拒绝启动并提示更换 `run-id` |
| `scripts/run_gold_audit.py` | **金标准审计**：独立抽检题库标准答案本身的准确性，排查源题库自身错标（与被测模型作答无关） |

### ③ 单元测试与回归门禁（Unit Tests & CI Gate）

| 检查套件 | 覆盖范围与当前状态 |
|---|---|
| `tests/` | **13 个模块 / 182 项断言**（168 个测试函数，含 14 项参数化展开；覆盖判分阶梯、配对检验、截断偏倚、留一法难度、乱序坐标还原、参数分流、断点续跑幂等性等） |
| `scripts/ci_gate.py` | **210 项**运行完整性与数据契约检查（有无 `raw/` 两种工作区状态下的完整读数说明见 [docs/reproduce.md](docs/reproduce.md)） |
| `scripts/exp_smoke.py` | **124 项**受控补充实验（E1～E6）冒烟断言，**124/124 全部通过** |

### ④ 报告数字逐格复算与渲染门禁（Report Verification）

| 脚本 | 核查内容 |
|---|---|
| `scripts/report_metrics.py` | **从 `tallies/` 判分台账逐格重算报告正文与表格里的每一个统计数字** |
| `scripts/_check_polished_report.py` · `_check_external_report.py` | 校验正文图序与附录图表索引一致、图片宽度合规、表格结构完整、内部 `§X.Y` 章节引用全部可解析 |
| `scripts/_check_render_safe.py` | **PDF 渲染安全兜底**：拦截反引号内含中文（CJK）、字体缺失符号与代码块含中文——此类字符在 XeLaTeX 编译 PDF 时会**静默渲染为空白且不报错** |

配合 [reports/report\_data\_provenance.md](reports/report_data_provenance.md)，**报告中的每一个图表与统计数字均可离线逐格复算**。

---

## 五、复现与核心文档导航

`runs/*/tallies/` 判分台账已随 Git 仓库入库，原始响应数据（`raw/`，约 75 MB）另通过 GitHub Release 数据包发布。**克隆仓库后无需配置任何 API Key，即可离线复算报告中的全部指标**——例如运行：

```powershell
.\.venv\Scripts\python.exe scripts\report_metrics.py --run-id c1-5arm-v2 --config c1
```

复算出的五个模型 MMLU-ProX 准确率与报告 §3.1 逐格吻合。每处统计数字的批次来源与复算命令均登记在 [reports/report\_data\_provenance.md](reports/report_data_provenance.md)。完整的环境搭建、跑批流程、门禁在两种工作区状态下的读数差异及目录结构说明，详见 [docs/reproduce.md](docs/reproduce.md)。

<details>
<summary>查看仓库核心文档与报告索引</summary>

| 路径 | 说明 |
|---|---|
| [reports/report\_external\_helm\_polished.md](reports/report_external_helm_polished.md) | **对外评测报告（唯一权威版）**：6 章结构，含完整方法学实证与分场景建议（配套同名 PDF） |
| [reports/PROJECT\_SUMMARY.md](reports/PROJECT_SUMMARY.md) | **评测工程闭环映射**：六大评测环节执行状态与全周期自我纠错记录 |
| [reports/report\_data\_provenance.md](reports/report_data_provenance.md) | **数据溯源清单**：全部表格、14 张图件与正文数字的来源批次及复算命令 |
| [reports/RUN\_AUDIT.md](reports/RUN_AUDIT.md) · [reports/RUN\_INVENTORY.md](reports/RUN_INVENTORY.md) | **批次审计台账**：全库 85 批次分类台账 · 37 个权威批次可用性核查 |
| [docs/design.md](docs/design.md) · [docs/param\_matrix.md](docs/param_matrix.md) | **方案设计与参数矩阵**：架构契约设计（含对 HELM 的三处修正）· 各模型参数生效实测表 |
| [docs/open\_issues.md](docs/open_issues.md) · [docs/limitations.md](docs/limitations.md) | **缺陷记录与结论边界**：全周期问题修复台账 · 统计与外推边界说明 |
| [docs/badcase\_method.md](docs/badcase_method.md) · [docs/reproduce.md](docs/reproduce.md) | **错题归因与复现指南**：LLM Judge 双次自一致率核查方案 · 环境搭建与门禁读数说明 |

</details>
