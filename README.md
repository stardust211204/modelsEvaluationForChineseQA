# modelsEvaluation

大模型**中文问答能力**的评测工程实践：以 HELM 的「场景 × 适配 × 指标」三元组为口径字典，自建轻量双管线框架，把「评测方案设计 → 评测集构建 → 判分体系 → 抽检复核 → badcase 归因 → 评测报告」这条闭环完整走一遍。

> **一句话定位**：**准确性是载体，评测闭环是目的** —— 本项目不是"再跑一遍榜单"，而是回答四个榜单不会回答的问题（见 §一）。
> **读结论前先读** [`docs/limitations.md`](docs/limitations.md)。
> **数字不手抄**：报告里每个数字都在 [`reports/report_data_provenance.md`](reports/report_data_provenance.md) 登记了来源与复算命令。

---

## 一、项目定位与成果物看板

### 四个榜单不会回答的问题

1. **判分口径本身会改变结论多少？** 同一批 680 条作答，严格口径 **0 条可判**（准确率**没有定义**，不是 0%），宽松口径恢复到 **40.6%（B 组）～93.4%（C 组）**。
2. **规则判分可信吗？** 五个模型各 150 条独立 judge 复核，一致率 99.3–100%、κ 0.986–1.000，未见同厂偏好。
3. **关闭思考条件下的排名，能代表模型能力吗？** 不能 —— 打开思考后五个模型一致上升 **25.0–36.8 个百分点**，且排名重洗。
4. **排名的可信边界在哪？** 前三名在 1:1 合并集上最大差 1.8pp，低于判定阈值；且两题集排序相反。**不给出总体排名。**

### 评测规模

| 项 | 值 |
|---|---|
| **被测模型** | 5 个：`deepseek-v4.1-flash`、`deepseek-v4-pro`、`longcat-2.5-preview-free`、`mimo-v2.6-flash`、`qwen3.8-flash` |
| **题集** | MMLU-ProX（588 → 剔除 42 道源缺陷题，计分 **546**）、C-Eval（**583**）、OR-Bench（**1,974** = Hard 1,319 + Toxic 655） |
| **评测任务** | **9 项**：直接作答 / 自由作答 / 作答格式递进 / 5-shot / 重复采样 / 选项乱序 / 选项标签替换 / 思考对照 / judge 复核 |
| **调用总量** | **53,379 次**（成功 53,291，失败 88），**37 个权威批次**（全库 85 批次 / 77,512 次） |
| **口径** | 准确率 = `correct ÷ (correct + wrong)`；`invalid` / `truncated` 不进分母、单独报告 |

### 成果物看板

| 我想看… | 去哪 |
|---|---|
| **评测本身做得怎么样** | [`reports/report_external_helm_polished.md`](reports/report_external_helm_polished.md) —— **唯一权威对外报告**（6 章 / 26 表 / 14 图 / 24 页 PDF） |
| **工程是怎么建的** | 本文件 §二 排障闭环 · §三 系统架构 · §四 质量防线 |
| **数字是真的吗** | [`reports/report_data_provenance.md`](reports/report_data_provenance.md) + `runs/*/tallies/`（判分台账**已入库**，clone 后可直接复算） |
| **批次怎么管的** | [`reports/RUN_AUDIT.md`](reports/RUN_AUDIT.md)（全库 85 批次分 A–F 六类）· [`reports/RUN_INVENTORY.md`](reports/RUN_INVENTORY.md)（37 权威批次选用 + 数据可用性核查） |
| **踩过什么坑** | 本文件 §二 · [`docs/open_issues.md`](docs/open_issues.md) |
| **还差什么** | 本文件 §五 已知局限 · [`docs/limitations.md`](docs/limitations.md) |
| **完整版报告** | `reports/_archive/report_external_helm_full.md`（35 表，含全部明细与推导；已冻结归档，仅供技术参考） |

---

## 二、六大评测陷阱与排障闭环

这一节是整个项目的**技术核心**：下面六类失真都是真实发生过的，每一类都配有「怎么发现的 → 怎么修 → 固化成哪条防线」。

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

## 三、系统架构与对 HELM 的三处工程修正

### 与 HELM 的关系

HELM 已于 2026-06-01 进入 maintenance mode，且其前提（开放权重 + logprob 打分）在封闭 API 环境不成立。
**该抄的**：分层架构、数据契约、截断与错误单列、污染显式声明。
**不能抄的**：logprob 打分路径、16 场景套件、多客户端抽象层、前端。

**三处修正**（`docs/design.md`）：

| 维度 | HELM | 本项目 | 理由 |
|---|---|---|---|
| 解析失败 | 并入错误，计为答错 | **独立 `invalid` 档**，不计入准确率分母 | 否则我们在测量一个被自己污染的量 |
| 生成截断 | 单独记录，**但仍计入准确率** | **独立 `truncated` 档**，不计入分母 | 推理模型把预算烧在思考上是常态；把"预算设小了"记成"模型答错"是错误的归因 |
| 多正确答案 | 原生支持 | **同样支持** `gold_indices: list[int]` | 否定式提问可能有争议答案，数据结构预留 |

### 五层管线与职责边界

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

**越界即为 bug —— 五条不可违反的纪律**：

1. **`execute` 层不解读模型输出。** 判分逻辑出现在这一层即为 bug。
2. **`tally` 层不发起任何模型调用。** 必须能对同一批 raw 反复重跑。
3. **few-shot 示例只能取自 `valid` split。** 取自 `test` 即数据泄漏。
4. **准确率只能有一个分母。** 分母是 `correct + wrong`，实现只允许有一份：`metrics.tier_accounting()`。
   历史上跑批日志自己算过 `correct / 全部`，于是同一批数据出现两套准确率 —— **两套分母共存比任何一个错值都危险**。
5. **统一题集规则**：截断率高时必须剔除「任一模型没做完」的题，且**在所有模型上剔同一批**。

### 核心工程决策：`raw` 与 `tally` 分离

```
runs/{run_id}/manifest.json                 # 参数生效表、代码版本、数据哈希
runs/{run_id}/raw/{config}/{arm}.jsonl      # 不可变：PromptSpec + ModelResult（不入库，走 Release）
runs/{run_id}/tallies/{config}/{arm}.jsonl  # 可重算：Tally（已入库，报告全部指标由它算出）
```

8 天窗口里最可能返工的是**判分逻辑**（解析阶梯、档位边界）。分离后改判分规则**不需要重跑 API** ——
项目期间判分口径改过多次，**一次 API 都没重跑**；返工成本从小时级降到秒级。

---

## 四、四道自动化质量防线

按测试生命周期排列。**这不是事后补的文档，是每次跑批实际执行的流程。**

### 防线 1 · 跑批前探针（Peak / Probe）

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

> 那 1 项 FAIL 是 `longcat` 的**上游连接失败**（间歇性路由缺陷，客户端无法修复），不是口径问题。
> 这里如实写 210 项而非「203 项全绿」：**门禁的价值在于它敢报红**。

### 防线 4 · 报告逐格自检

| 脚本 | 查什么 |
|---|---|
| `scripts/_check_polished_report.py` | **对外报告**：图序与图表索引一致、宽度沿用原报告、表格结构、章节引用可解析、不可渲染字符 |
| `scripts/_check_external_report.py` | 完整版报告的同类检查（**27/27**） |
| `scripts/_check_render_safe.py` | 反引号内含 CJK、字体缺失符号、代码块含中文 —— 这些在 PDF 里**静默渲染成空白，不报错** |
| `scripts/report_metrics.py` | 从 `tallies/` 逐格重算报告正文与表格数字 |

配合 [`reports/report_data_provenance.md`](reports/report_data_provenance.md)（35 表 / 14 图 / 52 处正文数字的来源与复算命令），做到**报告里没有一个手抄数字**。

---

## 五、核心发现与已知局限

### 五模型多维实测位置

| 模型 | 直接作答 · 思考关（合并 / MMLU-ProX / C-Eval） | 自由作答（MMLU-ProX） | 开启思考（76 题共同集，关 → 开） | 同题 10 次全同率 | 安全双向（过度拒答 / 危险服从） | 成本（Token / 中位延迟） |
|---|---|---|---|---|---|---|
| `longcat-2.5-preview-free` | **75.8%** / 62.9% / **88.0%** | 83.0%（+22.2） | 52.0 → 86.8%（+34.8） | **87.0%** | 54.7% / 3.5% | 64,439 / 9.60s |
| `deepseek-v4.1-flash` | 74.7% / **64.1%** / 84.6% | **83.5%**（+18.0） | 61.8 → 86.8%（+25.0） | 77.4% | 69.8% / 3.2% | 39,682 / 2.88s |
| `qwen3.8-flash` | 74.0% / 61.2% / 85.9% | 76.2%（+13.8） | 51.3 → 86.8%（+35.5） | 69.9% | **82.5% / 1.4%** | 144,322 / 12.78s |
| `deepseek-v4-pro` | 68.8% / 58.6% / 78.4% | 82.4%（+23.5） | 51.3 → **88.2%**（**+36.8**） | 80.6% | *判卷方未参评* | **30,113 / 2.87s** |
| `mimo-v2.6-flash` | 63.0% / 49.1% / 75.9% | 76.9%（**+29.1**） | 44.0 → 78.9%（+34.9） | **28.0%** | **46.6% / 8.6%** | 89,953 / 15.68s |

**各列实验条件不同，不能跨列比较绝对数值。** 完整表格、图件与统计检验见对外报告 §5.1。

### 六条核心发现

1. **判分口径的价值不是"制造伪低分"，而是"让作答从不可判定变成可判定"。** 同一模型、同一批题，只换口径就从「无法评分」变成「可以评分」（严格 0 条可判 → 宽松 40.6%~93.4%）。
2. **强格式约束本身有代价，量级 13.8–29.1pp**（配对显著）。机制不是"自由发挥更好"，而是强约束**拿掉了草稿纸与作答空间**。
3. **参数被忽略 ≠ 输出可复现。** 探针实测三个臂的温度**全部被忽略**，但同题 10 次仍有模型只有 **28.0%** 的题答案全同。⇒「固定温度 = 可复现」不成立；单次评测自带重复噪声，而这条信息**只有重复采样能量化**。
4. **思考开关是效应最大的单一变量**，且会改变排名。任何跨模型对比都必须声明思考状态。
5. **单一总分不足以区分这五个模型。** 前三名差距低于判定阈值、两集排序相反；而分领域、稳定性、安全三个维度上差异显著。
6. **示例的示范作用强于正文指令 —— 一次真实的实验设计缺陷。** 5-shot 首轮只因示例写成 `答案：X`，就造出了「+11.9~+17.3pp、统计显著」这个方向明确、统计显著、**且完全错误**的结论。**改了 prompt，第一件事是把渲染结果打印出来人眼过一遍，而不是直接跑 750 次。**

### 已知局限

- **badcase 归因由 LLM judge 完成，未做人工复核。** 归因逻辑已实现并跑完（`scripts/run_badcase_judge.py`）：**99/104** 条错题可标，用 `deepseek-v4-pro`（关思考、temp 0.1）**跑两遍**测自一致率，得 **93.9%（κ = 0.914）** —— 这是标签稳定性的证据，**不是人工一致率**。因此归因分布属于「单标注者 + 自一致率佐证」的强度，**不作为强结论使用**。标准答案审计独立进行（`scripts/run_gold_audit.py`）：61 题中 **56 题（91.8%）无缺陷**。
  > ⚠️ **一条待查线索**：人工标注的 20 条子集中有 **9 条判为 `gold_error`（标准答案有误）**。该子集只覆盖单个臂、样本量不足（方法文档要求 ≥40 条），**不能当作总体比例**；但 `docs/badcase_method.md` 明确「`gold_error` ≥3% ⇒ 需修正已发布准确率」，故如实记录，留作后续核查项。
- `manifest.params_effective` 有 13 处与实际请求参数不符（3 处在权威批次上）。**报告结论不受影响** —— raw 记录才是真值。见 [`reports/RUN_INVENTORY.md`](reports/RUN_INVENTORY.md) §4.6。
- 完整局限见 [`docs/limitations.md`](docs/limitations.md)。

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
> 五个模型的 MMLU-ProX 准确率复算为 58.6 / 64.1 / 62.9 / 49.1 / 61.2（n=546），
> 与报告 §3.1 逐个吻合。

> **门禁的两种读数**（差别只在 raw 在不在）：
>
> | 场景 | `ci_gate`（210 项） | `exp_smoke`（124 项） |
> | --- | --- | --- |
> | **完整工作区**（raw 已下载） | PASS **203** / FAIL **1** / WARN 6 | **124/124** |
> | **刚 clone**（raw 走 Release，不在 git 里） | PASS 162 / FAIL **0** / **SKIP 42** | **123/123**（跳过 1 项） |
>
> 被跳过的是**输入不存在**的检查（「raw 完整性」「judge tally 完整性」「被复核 run 含 5 臂」等），
> 脚本会写明原因 —— **是跳过，不是失败**。
> 那 1 项 FAIL 是 `longcat` 的**上游连接失败**（间歇性路由缺陷，客户端无法修复），不是口径问题。
> 这里如实写 210 项而非「203 项全绿」：**门禁的价值在于它敢报红**。
>
> 判据是**全局**的：只要有一个 `runs/*/raw` 存在，就按完整工作区标准判 ——
> 此时某个批次缺 raw 是**真事故**，照报 FAIL。要拿到完整读数，
> 下载 Release 里的 raw 包解压到仓库根即可（见 [`docs/reports-fix.md`](docs/reports-fix.md) §10.4）。

**小样本走完整条管线**：`.\run_all.ps1`（末步即门禁）。

**报告排版**：PDF 走 `build_pdf.py`（pandoc + xelatex + Eisvogel），排版参数取自 **CCL 2026 官方中文论文模板** —— 宋体正文 + Times New Roman 西文 + 单倍行距 + 首行缩进 1em + 一级标题 12pt 黑体 / 二三级 10pt 黑体。调研产物（含官方模板原件与编译预览）在 `reports/_verify/web/`。

### 目录结构

```
src/meval/       五层管线本体（见 §三）
scripts/         入口、实验仪器与核校工具（~190 个）
configs/         arms.yaml（五臂 + 各自生效参数）· matrix.yaml · exclusions.json
data/            eval_set.jsonl · ceval_set.jsonl · orbench_set.jsonl · frozen.json
prompts/         c1_strict / c2_free / c5_fewshot / e2_labelshuffle / e6_* / judge_*
docs/            design.md · limitations.md · open_issues.md · param_matrix.md · badcase_method.md
figures/         14 张正式图 + latex/（管线图源码与 HOWTO.md）
runs/            每次跑批：manifest + raw（未入库） + tallies（已入库，可直接复算）
reports/         对外报告 · 溯源清单 · 批次台账 · 专项实验报告
tests/           13 个模块 / 168 个用例
```
