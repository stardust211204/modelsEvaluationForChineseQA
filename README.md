# modelsEvaluation

**大模型中文问答能力的评测工程实践** —— 以 HELM 的「场景 × 适配 × 指标」三元组为口径字典，自建轻量双管线框架，覆盖「评测方案设计 → 评测集构建 → 判分体系 → 抽检复核 → badcase 归因 → 评测报告」六个环节。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/%E5%8D%95%E6%B5%8B-168%20passed-brightgreen.svg)](tests/)

> [评测规模](#一评测规模) · [系统架构](#二系统架构与对-helm-的三处工程修正) · [项目中的缺陷与经验](#三项目中的缺陷与经验) · [四层自动化校验](#四四层自动化校验) · [复现](#五复现) · [成果物映射](reports/PROJECT_SUMMARY.md)

---

## 一、评测规模

| 项 | 值 |
|---|---|
| **被测模型** | 5 个：`deepseek-v4.1-flash`、`deepseek-v4-pro`、`longcat-2.5-preview-free`、`mimo-v2.6-flash`、`qwen3.8-flash` |
| **题集** | MMLU-ProX（588 → 剔除 42 道源缺陷题，计分 **546**）、C-Eval（**583**）、OR-Bench（**1,974** = Hard 1,319 + Toxic 655） |
| **评测任务** | **9 项**：直接作答 / 自由作答 / 作答格式递进 / 5-shot / 重复采样 / 选项乱序 / 选项标签替换 / 思考对照 / judge 复核 |
| **调用总量** | **53,379 次**（成功 53,291，失败 88），**37 个权威批次**（全库 85 批次 / 77,512 次） |
| **口径** | 准确率 = `correct ÷ (correct + wrong)`；`invalid` / `truncated` 不进分母、单独报告 |

> 结论与统计检验见 [对外报告](reports/report_external_helm_polished.md)（§1.3 实测速览 → 第 4 章 → §5.1 方法学结论）。

---

## 二、系统架构与对 HELM 的三处工程修正

### 与 HELM 的关系

HELM 已于 2026-06-01 进入 maintenance mode，且其前提（开放权重 + logprob 打分）在封闭 API 环境不成立。
**本项目采用**：分层架构、数据契约、截断与错误单列、污染显式声明。
**本项目未采用**：logprob 打分路径、16 场景套件、多客户端抽象层、前端。

**三处修正**（`docs/design.md`）：

| 维度 | HELM | 本项目 | 理由 |
|---|---|---|---|
| 解析失败 | 并入错误，计为答错 | **独立 `invalid` 档**，不计入准确率分母 | 否则会把判分口径自身的问题计入模型表现 |
| 生成截断 | 单独记录，**但仍计入准确率** | **独立 `truncated` 档**，不计入分母 | 推理模型把推理预算耗在思考上是常态；把"预算设小了"记成"模型答错"是错误的归因 |
| 多正确答案 | 原生支持 | **同样支持** `gold_indices: list[int]` | 否定式提问可能有争议答案，数据结构预留 |

### 五层管线与职责边界

![准确率评测管线的五层划分](figures/40_pipeline_main.png)

<details>
<summary>五层模块清单</summary>

```
src/meval/
  types.py       数据模型
  build_set.py   ① 造题：分层抽样 + 冻结哈希
  adapt.py       ② 请求构造（纯函数 + 哨兵插槽）
  execute.py     ③ 执行：并发 / 退避 / 断点续跑 / 原始落盘
  tally.py       ④ 判分：严格·宽松双口径 + 四档判级
  metrics.py     ⑤ 聚合：四档分账 / McNemar / Kappa / 留一法难度
  client.py      OpenAI 兼容客户端（参数分流）
  exclusions.py  题集剔除清单（42 题，带指纹）
```

</details>

### 各层职责划分与架构契约

管线按单向依赖分层。以下五条约束保证实验结果可复现、可审计：

1. **`execute` 层只管调用与落盘**：负责网络并发、退避重试与原始响应持久化，不含任何解析或判分逻辑 —— 这样判分规则怎么演进都不影响采集。
2. **`tally` 层是纯离线计算**：只消费已落盘的 raw，不发起任何模型调用 —— 同一批原始响应可零成本反复重算。
3. **few-shot 示例只取自 `valid` split**：物理上不读 `test` split，杜绝数据泄漏。
4. **准确率只允许一个分母**：分母恒为 `correct + wrong`，结算收口在 `metrics.tier_accounting()` 一处，跑批日志与图表脚本都不得另算一份。
5. **跨模型对比用统一共同集**：截断率高时强制收敛到所有模型共同完成的交集，且在所有模型上剔同一批。

> 第 4 条防的是**分母漂移**：一旦下游日志或绘图脚本各自另算一份，同一批数据就会出现两套准确率 —— **两套分母共存比单个错值更难发现**，两个数都「有来源」，谁也不会觉得它错。

### 核心工程决策：`raw` 与 `tally` 分离

最可能返工的是**判分逻辑**（解析阶梯、档位边界）。把不可变的模型原始输出与可重算的判分结果分开存放后，
改判分规则**不需要重跑 API** —— 项目期间判分口径改过多次，**一次 API 都没重跑**；返工成本从小时级降到秒级。

<details>
<summary>目录布局与两者分工</summary>

```
runs/{run_id}/manifest.json                 # 参数生效表、代码版本、数据哈希
runs/{run_id}/raw/{config}/{arm}.jsonl      # 不可变：PromptSpec + ModelResult（不入库，走 Release）
runs/{run_id}/tallies/{config}/{arm}.jsonl  # 可重算：Tally（已入库，报告全部指标由它算出）
```

</details>

---

## 三、项目中的缺陷与经验

以下六条是跑批中实际踩到的缺陷，以及最终固化进代码的修法。它们大多属于**黑盒 API 场景下不易察觉**的那一类，因此每一条都留了测试或配置作为防线。

| # | 缺陷 | 代价 | 经验与修法 | 固化为 |
|---|---|---|---|---|
| **1** | 推理模型的**静默失败** | 推理预算耗尽时返回空串，**而 HTTP 状态是 200**。这类题本身偏难（正确率比未截断题低 **28 个百分点**），直接剔除会让结果虚高 | 按 `finish_reason` 把截断独立归入 `truncated`，跨模型对比时强制收敛到五模型共同完成的题集 | `tests/test_truncation_bias.py` · `tests/test_tally.py` |
| **2** | **裁判模型的思考没关** | 隐藏推理吃光输出预算，约 **9%** 的判定拿不到结果，且失败集中在本身易错的题上 —— 一致率会被高估 | 判分前显式冻结 judge 的 `reasoning_effort="none"`，并记入运行元数据 | `scripts/_check_judge_verdicts.py` |
| **3** | **示例的示范作用压倒正文指令**；跨集选项数错配 | 首轮示例写成 `答案：X`，模型照抄这个外形，**87%** 的输出无法严格解析 —— 当时算出的那轮「增益」其实来自格式顺从。另有 C-Eval 的 4 选项题沿用了 10 选项示例 | 示例改为示范目标输出格式本身；选项数量严格对齐；模板改动前做指纹比对，不一致即拒绝跑批 | `tests/test_prompt_letters.py` · `src/meval/adapt.py` |
| **4** | **分母不一致时直接相减** | 放宽格式后各档的可判条目数差一倍以上（**276 对 635**），直接相减会得出方向相反的结论 | 跨配置比较一律限制在双方均可判的**配对交集**上，做配对 McNemar 检验 | `tests/test_paired.py` · `scripts/exp_e6_paired.py` |
| **5** | **两个指标其实不可比**；乱序未还原坐标系 | 「是否翻转过」随观测次数单调上升，观测次数不同的实验不能直接相除；选项乱序若不还原原题坐标系，会算出虚假翻转 | 改用与次数无关的「逐对不一致率」；乱序后强制还原坐标系再比 | `tests/test_position_refusal.py` · `scripts/_ratio_pairwise.py` |
| **6** | **温度设了也不等于可复现**；厂商私有参数冲突 | 探针实测部分厂商直接忽略温度参数；而即便忽略，同题重复 10 次的全同率最低也只有 **28%** | 放弃全局统一参数，接入新模型前先跑探针，把各臂生效的参数冻结进配置；私有参数走 `extra_body` | `tests/test_client_params.py` · `docs/param_matrix.md` |

---

## 四、四层自动化校验

按测试生命周期排列。

### ① 跑批前探针

| 脚本 | 查什么 |
|---|---|
| `scripts/smoke_arms.py` | 五臂连通性与最小可用性 |
| `scripts/probe_params.py` | 各臂的**参数生效边界**（温度是否有效、思考怎么关、`max_tokens` 下限） |
| `scripts/probe_distinct.py` | **伪别名行为指纹** —— `resp_model` 只回显请求名，故用 30 题 × 各臂的答案一致率反查是否同一后端 |
| `scripts/compare_endpoints.py` | 同模型不同端点的一致性 |
| `scripts/probe_alias.py` / `probe_new_arm.py` | 新增臂接入前的定点探测 |

结果回填 `docs/param_matrix.md`，**冻结**进 `configs/arms.yaml` —— 不做全局统一参数，因为实测证明参数跨厂商不可约。

### ② 数据清洗与指纹冻结

| 机制 | 说明 |
|---|---|
| `configs/exclusions.json` + `src/meval/exclusions.py` | **42 道源缺陷题**（37 题 LaTeX 公式损坏 + 5 题选项重复），题集不动、只记一份「不可用于统计」的清单。判据在 `scripts/_check_broken_latex.py`（`\命令` 是否真实存在），生成脚本 `scripts/build_exclusions.py` |
| `data/frozen.json` | 题集与提示词的冻结指纹；`scripts/_replay_fingerprints.py` 可重放校验 |
| `src/meval/adapt.py::prompt_fingerprint()` | 模板指纹校验：`scripts/run_config.py` 跑批前比对指纹，不一致即拒绝启动并提示换 run-id |
| `scripts/run_gold_audit.py` | 金标准审计（标准答案本身有没有缺陷，与模型作答无关） |

### ③ 单元测试与回归门禁

| 项 | 当前状态 |
|---|---|
| `tests/` | **13 个模块 / 168 个用例**（判分阶梯、配对检验、截断偏倚、难度、位置还原、参数分流、断点续跑……） |
| `scripts/ci_gate.py` | **210 项**运行完整性与数据契约检查（两种工作区状态下的完整读数见 [`docs/reproduce.md`](docs/reproduce.md)） |
| `scripts/exp_smoke.py` | **124 项**补充实验冒烟，**124/124** |

### ④ 报告数字逐格复算

| 脚本 | 查什么 |
|---|---|
| `scripts/report_metrics.py` | **从 `tallies/` 逐格重算报告正文与表格里的每一个数字** |
| `scripts/_check_polished_report.py` · `_check_external_report.py` | 图序与图表索引一致、表格结构、章节引用可解析（完整版 **27/27**） |
| `scripts/_check_render_safe.py` | 反引号内含 CJK、字体缺失符号、代码块含中文 —— 这些在 PDF 里**静默渲染成空白，不报错**，属排版侧兜底 |

配合 [`reports/report_data_provenance.md`](reports/report_data_provenance.md)，**报告中的数字均可逐格复算**。

---

## 五、复现

`runs/*/tallies/` 判分台账已随仓库入库，原始响应（约 75 MB）另以 Release 数据包发布。
**clone 后无需任何 API key 即可复算报告里的全部指标** —— 例如

```powershell
.\.venv\Scripts\python.exe scripts\report_metrics.py --run-id c1-5arm-v2 --config c1
```

复算出的五模型 MMLU-ProX 准确率与报告 §3.1 逐格吻合。每处数字的来源与复算命令登记在
[`reports/report_data_provenance.md`](reports/report_data_provenance.md)。

环境搭建、跑批流程、门禁在两种工作区状态下的读数差异、以及目录结构，见
[`docs/reproduce.md`](docs/reproduce.md)。
