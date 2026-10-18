# modelsEvaluation

**大模型中文问答能力的评测工程实践** —— 以 HELM 的「场景 × 适配 × 指标」三元组为口径字典，自建轻量双管线框架，把「评测方案设计 → 评测集构建 → 判分体系 → 抽检复核 → badcase 归因 → 评测报告」这条闭环完整走一遍。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![Runs](https://img.shields.io/badge/%E6%9D%83%E5%A8%81%E6%89%B9%E6%AC%A1-37%20%2F%2085-blue.svg)](reports/RUN_AUDIT.md)
[![Calls](https://img.shields.io/badge/API%20%E8%B0%83%E7%94%A8-53%2C379-blue.svg)](reports/RUN_AUDIT.md)
[![Tests](https://img.shields.io/badge/%E5%8D%95%E6%B5%8B-168%20passed-brightgreen.svg)](tests/)
[![Report](https://img.shields.io/badge/%E6%8A%A5%E5%91%8A-27%20%E8%A1%A8%20%2F%2014%20%E5%9B%BE-purple.svg)](reports/report_external_helm_polished.md)

> - **一句话定位**：**准确性是载体，评测工程闭环是目的** —— 本项目侧重用单因子受控实验量化评测协议自身的失真边界。
> - **工程与质量基线**：五层解耦双管线 · 37 个权威批次 / 53,379 次调用 · 13 模块 168 项单测 · 210 项 CI 回归门禁 · 报告全部数字无需 API 离线逐格复算。
> - **快速导航**：[评测规模](#一项目定位与评测规模) · [系统架构](#二系统架构与对-helm-的三处工程修正) · [六大排障陷阱](#三六大评测陷阱与排障闭环) · [四道质量防线](#四四道自动化质量防线) · [快速复算](#六快速开始与一键复算)
> - **要按图索骥找文件**：见 [`reports/PROJECT_SUMMARY.md`](reports/PROJECT_SUMMARY.md) —— 它按「每一环 → 产出什么 → 怎么验证」给出完整映射。

---

## 一、项目定位与评测规模

| 项 | 值 |
|---|---|
| **被测模型** | 5 个：`deepseek-v4.1-flash`、`deepseek-v4-pro`、`longcat-2.5-preview-free`、`mimo-v2.6-flash`、`qwen3.8-flash` |
| **题集** | MMLU-ProX（588 → 剔除 42 道源缺陷题，计分 **546**）、C-Eval（**583**）、OR-Bench（**1,974** = Hard 1,319 + Toxic 655） |
| **评测任务** | **9 项**：直接作答 / 自由作答 / 作答格式递进 / 5-shot / 重复采样 / 选项乱序 / 选项标签替换 / 思考对照 / judge 复核 |
| **调用总量** | **53,379 次**（成功 53,291，失败 88），**37 个权威批次**（全库 85 批次 / 77,512 次） |
| **口径** | 准确率 = `correct ÷ (correct + wrong)`；`invalid` / `truncated` 不进分母、单独报告 |

> **结论、得分表与统计检验都在对外报告里**，本文件不重复：
> [`reports/report_external_helm_polished.md`](reports/report_external_helm_polished.md)（6 章 / 27 表 / 14 图。
> 建议读序：§1.3 实测速览 → 第 4 章 → §5.1 方法学结论）。

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
| 生成截断 | 单独记录，**但仍计入准确率** | **独立 `truncated` 档**，不计入分母 | 推理模型把预算烧在思考上是常态；把"预算设小了"记成"模型答错"是错误的归因 |
| 多正确答案 | 原生支持 | **同样支持** `gold_indices: list[int]` | 否定式提问可能有争议答案，数据结构预留 |

### 五层管线与职责边界

![准确率评测管线的五层划分](figures/40_pipeline_main.png)

<details>
<summary>五层模块清单（点击展开）</summary>

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

**各层职责边界（越界即为缺陷）—— 五条纪律**：

1. **`execute` 层不解读模型输出。** 判分逻辑出现在这一层即为 bug。
2. **`tally` 层不发起任何模型调用。** 必须能对同一批 raw 反复重跑。
3. **few-shot 示例只能取自 `valid` split。** 取自 `test` 即数据泄漏。
4. **准确率只能有一个分母。** 分母是 `correct + wrong`，实现只允许有一份：`metrics.tier_accounting()`。
   历史上跑批日志自己算过 `correct / 全部`，于是同一批数据出现两套准确率 —— **两套分母共存比单个错值更难发现**。
5. **统一题集规则**：截断率高时必须剔除「任一模型没做完」的题，且**在所有模型上剔同一批**。

### 核心工程决策：`raw` 与 `tally` 分离

最可能返工的是**判分逻辑**（解析阶梯、档位边界）。把不可变的模型原始输出与可重算的判分结果分开存放后，
改判分规则**不需要重跑 API** —— 项目期间判分口径改过多次，**一次 API 都没重跑**；返工成本从小时级降到秒级。

<details>
<summary>目录布局与两者分工（点击展开）</summary>

```
runs/{run_id}/manifest.json                 # 参数生效表、代码版本、数据哈希
runs/{run_id}/raw/{config}/{arm}.jsonl      # 不可变：PromptSpec + ModelResult（不入库，走 Release）
runs/{run_id}/tallies/{config}/{arm}.jsonl  # 可重算：Tally（已入库，报告全部指标由它算出）
```

</details>

---

## 三、六大评测陷阱与排障闭环

本节记录六类实际发生过的失真，以及各自的发现方式、处理办法与固化成的防线。

| # | 陷阱 | 现象与实测影响 | 应对 | 固化为 |
|---|---|---|---|---|
| **1** | **推理模型空 `content` 静默失败与截断幸存者偏差** | 预算烧在内部思维链上返回空串，**而 HTTP 状态是 200**。1,024 预算下 **31% 截断**、且截断题系统性偏难（截断题 65.4% vs 未截断题 93.5%，差 **28.1pp**），直接剔除使准确率**虚高 7.8pp** | `finish_reason=='length'` 独立归入 `truncated`；扩至 8,192 预算并强制取五模型 **76 题统一共同集** | `tests/test_truncation_bias.py` · `tests/test_tally.py` |
| **2** | **LLM Judge 思考未关 = 选择性偏差**（**三臂期**） | C4 首轮 judge 隐藏思考吃光预算，**9.1%（41/450）**判不出，**且失败集中在规则判错的题上** → 一致率被高估 | 判分前显式冻结 judge `reasoning_effort="none"`，并记入 manifest；覆盖率 85–92% → **98%~100%** | `scripts/_check_judge_verdicts.py` |
| **3** | **Few-shot 示例格式压倒正文指令 & 跨集选项数错配**（**三臂期**） | C5 首轮示例写成 `答案：X`，伪造出 **+11.9~+17.3pp** 的假显著增益（严格口径 393/450 判不出）；C-Eval 共用 10 选项示例致 `deepseek-v4-pro` 暴跌 16.0pp | 示例强制示范**目标输出格式本身**、严取 `valid` split，并按目标题选项数严格匹配 | `tests/test_prompt_letters.py` · `adapt.prompt_fingerprint()` 守卫（改模板即拒跑） |
| **4** | **自由作答拆解分母不一致 → 因果归因反转** | B 组宽松口径仅 **276/680** 可判，直接拿各组绝对准确率相减会得出「末行要求贡献最大（+13.8 vs +8.1）」的**相反结论** | 跨配置比较一律限制在双方均可判的**配对交集**上做 McNemar；修正后为「允许写理由（**+15.9**）> 末行要求（**+10.0**）」 | `tests/test_paired.py` · `scripts/exp_e6_paired.py` |
| **5** | **跨采样次数翻转率不可除 & 乱序未还原坐标系** | 选项重排（4 次）与重复采样（10 次）观测次数不同，翻转率随次数单调上升使比值被压低；C-Eval 乱序未还原坐标系会产生 **67%~76%** 假值 | 改用与观测次数无关的**逐对不一致率**；强制还原原题坐标系再比 | `tests/test_position_refusal.py` · `scripts/_ratio_pairwise.py` |
| **6** | **固定温度 ≠ 输出可复现 & 厂商私有参数冲突** | 探针实测多臂**忽略温度参数**，但同题重复 10 次全同率最低仅 **28.0%**；不同厂商关闭思考的参数互斥，或本地抛 `TypeError` | 放弃全局统一参数，改为按臂探测后冻结至 `configs/arms.yaml`；私有参数走 `extra_body` | `tests/test_client_params.py` · `docs/param_matrix.md` |

> 陷阱 2、3 标注「三臂期」是因为它们发生在项目早期的三臂阶段。数字与结论未经修改，但口径范围要如实说明。

---

## 四、四道自动化质量防线

按测试生命周期排列。**四道防线都是跑批流程中的实际环节，各有对应脚本与失败记录。**

### 防线 1 · 跑批前探针

| 脚本 | 查什么 |
|---|---|
| `scripts/smoke_arms.py` | 五臂连通性与最小可用性 |
| `scripts/probe_params.py` | 各臂的**参数生效边界**（温度是否有效、思考怎么关、`max_tokens` 下限） |
| `scripts/probe_distinct.py` | **伪别名行为指纹** —— `resp_model` 只回显请求名，故用 30 题 × 各臂的答案一致率反查是否同一后端 |
| `scripts/compare_endpoints.py` | 同模型不同端点的一致性 |
| `scripts/probe_alias.py` / `probe_new_arm.py` | 新增臂接入前的定点探测 |

结果回填 `docs/param_matrix.md`，**冻结**进 `configs/arms.yaml` —— 不做全局统一参数，因为实测证明参数跨厂商不可约。

### 防线 2 · 数据清洗与指纹冻结

| 机制 | 说明 |
|---|---|
| `configs/exclusions.json` + `src/meval/exclusions.py` | **42 道源缺陷题**（37 题 LaTeX 公式损坏 + 5 题选项重复），题集不动、只记一份「不可用于统计」的清单。判据在 `scripts/_check_broken_latex.py`（`\命令` 是否真实存在），生成脚本 `scripts/build_exclusions.py` |
| `data/frozen.json` | 题集与提示词的冻结指纹；`scripts/_replay_fingerprints.py` 可重放校验 |
| `src/meval/adapt.py::prompt_fingerprint()` | **改模板即拒跑**：`scripts/run_config.py` 跑批前比对指纹，不一致就拒绝启动并提示换 run-id |
| `scripts/run_gold_audit.py` | 金标准审计（标准答案本身有没有缺陷，与模型作答无关） |

### 防线 3 · 单元测试与回归门禁

| 项 | 当前状态 |
|---|---|
| `tests/` | **13 个模块 / 168 个用例**（判分阶梯、配对检验、截断偏倚、难度、位置还原、参数分流、断点续跑……） |
| `scripts/ci_gate.py` | **210 项**运行完整性与数据契约检查 —— **PASS 203 / FAIL 1 / WARN 6** |
| `scripts/exp_smoke.py` | **124 项**补充实验冒烟，**124/124** |

> 那 1 项 FAIL 是 `longcat` 的**上游连接失败**（间歇性路由缺陷，客户端无法修复），与口径无关。
> 这里如实写「210 项，PASS 203 / FAIL 1」：**门禁的意义在于它报告真实状态，包括未通过项**。

### 防线 4 · 报告逐格自检

| 脚本 | 查什么 |
|---|---|
| `scripts/_check_polished_report.py` | **对外报告**：图序与图表索引一致、宽度沿用原报告、表格结构、章节引用可解析、不可渲染字符 |
| `scripts/_check_external_report.py` | 完整版报告的同类检查（**27/27**） |
| `scripts/_check_render_safe.py` | 反引号内含 CJK、字体缺失符号、代码块含中文 —— 这些在 PDF 里**静默渲染成空白，不报错** |
| `scripts/report_metrics.py` | 从 `tallies/` 逐格重算报告正文与表格数字 |

配合 [`reports/report_data_provenance.md`](reports/report_data_provenance.md)（35 表 / 14 图 / 52 处正文数字的来源与复算命令），**报告中的数字均可逐格复算**。

---

## 五、已知局限与工程边界

- **badcase 归因由 LLM judge 完成，未做人工复核。** 归因逻辑已实现并跑完（`scripts/run_badcase_judge.py`）：**99/104** 条错题可标，用 `deepseek-v4-pro`（关思考、temp 0.1）**跑两遍**测自一致率，得 **93.9%（κ = 0.914）** —— 这是标签稳定性的证据，**不等于人工一致率**。因此归因分布属于「单标注者 + 自一致率佐证」的强度，**不作为强结论使用**。标准答案审计独立进行（`scripts/run_gold_audit.py`）：61 题中 **56 题（91.8%）无缺陷**。
  > 其中一条待查线索的抽样背景较易被误读（20 条子集取自「多模型共错」的高危候选池，比例不能外推到全库），完整说明见 [`docs/limitations.md`](docs/limitations.md) 的 badcase 归因条目。
- 完整局限、`manifest.params_effective` 的 13 处登记不符等细节，见 [`docs/limitations.md`](docs/limitations.md) 与 [`reports/RUN_INVENTORY.md`](reports/RUN_INVENTORY.md) §4.6。

---

## 六、快速开始与一键复算

```powershell
# 1) 环境（不要复制别人的 .venv —— 里面有绝对路径）
uv venv .venv --python 3.12
uv pip install --python ".\.venv\Scripts\python.exe" `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

# 2) 密钥
Copy-Item .env.example .env    # 填入真实 key

# 3) 跑批前探针（★ 必做，结果回填 docs/param_matrix.md）
.\.venv\Scripts\python.exe scripts\smoke_arms.py
.\.venv\Scripts\python.exe scripts\probe_params.py

# 4) 单测 + 两个门禁
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\ci_gate.py      # 210 项
.\.venv\Scripts\python.exe scripts\exp_smoke.py    # 124 项

# 5) 出报告（Markdown → PDF；默认源即唯一权威报告）
.\.venv\Scripts\python.exe scripts\build_pdf.py

# 6) 报告自检与逐格复算（不需要 API）
.\.venv\Scripts\python.exe scripts\_check_polished_report.py
.\.venv\Scripts\python.exe scripts\_check_render_safe.py
.\.venv\Scripts\python.exe scripts\report_metrics.py --run-id c1-5arm-v2 --config c1
```

> **只复算、不跑批**：第 4~6 步全部是离线脚本。`runs/*/tallies/` 判分台账**已入库**，
> clone 后**无需任何 API key** 即可复算报告全部指标。
> 实测：全新 clone 里跑 `report_metrics.py --run-id c1-5arm-v2 --config c1`，
> 五个模型的 MMLU-ProX 准确率复算为 `deepseek-v4-pro` 58.6 / `deepseek-v4.1-flash` 64.1 / `longcat-2.5` 62.9 / `mimo-v2.6` 49.1 / `qwen3.8` 61.2（n=546），
> 与报告 §3.1 逐个吻合。

<details>
<summary>门禁在两种工作区状态下的读数差异（点击展开）</summary>

> 差别只在 `raw` 在不在（原始响应包走 Release，不入库）：

| 场景 | `ci_gate`（210 项） | `exp_smoke`（124 项） |
| --- | --- | --- |
| **完整工作区**（raw 已下载） | PASS **203** / FAIL **1** / WARN 6 | **124/124** |
| **刚 clone**（raw 未下载） | PASS 162 / FAIL **0** / **SKIP 42** | **123/123**（跳过 1 项） |

> 被跳过的是**输入不存在**的检查（「raw 完整性」「judge tally 完整性」「被复核 run 含 5 臂」等），
> 脚本会写明原因 —— **这些是跳过项，不计为失败**（§四 防线 3 另说明了那 1 项 FAIL 的来历）。
>
> 判据是**全局**的：只要有一个 `runs/*/raw` 存在，就按完整工作区标准判 ——
> 此时某个批次缺 raw 是**真事故**，照报 FAIL。要拿到完整读数，
> 把 Release 里的 raw 包解压到仓库根（保持 `runs/{run_id}/raw/...` 结构）即可。

</details>

**小样本走完整条管线**：`.\run_all.ps1`（末步即门禁）。**报告排版**：`build_pdf.py` 一键编译 PDF（pandoc + xelatex + Eisvogel），参数取自 CCL 2026 官方中文论文模板。

<details>
<summary>目录结构（点击展开）</summary>

```
src/meval/       五层管线本体（见 §二）
scripts/         入口、实验与分析脚本、核校工具（~190 个）
configs/         arms.yaml（五臂 + 各自生效参数）· matrix.yaml · exclusions.json
data/            eval_set.jsonl · ceval_set.jsonl · orbench_set.jsonl · frozen.json
prompts/         c1_strict / c2_free / c5_fewshot / e2_labelshuffle / e6_* / judge_*
docs/            design.md · limitations.md · open_issues.md · param_matrix.md · badcase_method.md
figures/         14 张正式图 + latex/（管线图源码与 HOWTO.md）
runs/            每次跑批：manifest + raw（未入库） + tallies（已入库，可直接复算）
reports/         对外报告 · 溯源清单 · 批次台账 · 专项实验报告
tests/           13 个模块 / 168 个用例
```

</details>
