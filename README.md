# modelsEvaluation

大模型**中文问答能力**的竞品评测。以 HELM 的「场景 × 适配 × 指标」三元组为口径字典，自建轻量管线，把「评测方案设计 → 评测集构建 → 判分体系 → 抽检复核 → 稳定性与位置敏感性 → 安全评测 → 竞品报告」这条评测工程闭环走完一遍。

> **定位**：作品集项目。五个模型经第三方聚合端点接入，结论用于**业务层竞品分析**，不宣称权威排名。
> **读结论前先读**：[`docs/limitations.md`](docs/limitations.md)。不读 Limitations 的结论不要引用。
> **已知缺陷**：[`docs/open_issues.md`](docs/open_issues.md)。
> **数字不手抄**：报告里的每个数字都在 [`reports/report_data_provenance.md`](reports/report_data_provenance.md) 里登记了来源与复算命令。

---

## 一、这个项目在做什么

**以准确性评测为载体，把「评测方案设计 → 评测集构建 → 判分体系 → 抽检复核 → badcase 归因 → 评测报告」这条评测工程闭环完整走一遍**（定位见 [`docs/design.md`](docs/design.md)）。准确性是载体，闭环是目的 —— 所以本项目不是"再跑一遍榜单"，而是回答四个榜单不会回答的问题：

1. **判分口径本身会改变结论多少？** 同一批 680 条作答，严格口径 **0 条可判**（准确率没有定义，不是 0%），宽松口径恢复到 **40.6%（B 组）～93.4%（C 组）**。
2. **规则判分可信吗？** 五个模型各 150 条独立 judge 复核，一致率 99.3–100%、κ 0.986–1.000。
3. **关闭思考条件下的排名，能代表模型能力吗？** 不能 —— 打开思考后五个模型一致上升 **25.0–36.8 个百分点**，且排名重洗。
4. **排名的可信边界在哪？** 前三名在 1:1 合并集上最大差 1.8pp，低于判定阈值；且两题集排序相反。**本报告不给出总体排名。**

对 HELM 的三处关键修正：解析失败独立成档、生成截断独立成档、支持一题多正解。

---

## 二、评测规模

| 项 | 值 |
|---|---|
| **被测模型** | 5 个：`deepseek-v4.1-flash`、`deepseek-v4-pro`、`longcat-2.5-preview-free`、`mimo-v2.6-flash`、`qwen3.8-flash` |
| **题集** | MMLU-ProX（588 题 → 剔除 42 道源缺陷题，计分 **546**）、C-Eval（**583**，无剔除）、OR-Bench（**1,974** = Hard 1,319 + Toxic 655） |
| **评测任务** | **9 项**：直接作答 / 自由作答 / 作答格式递进 / 5-shot / 重复采样 / 选项乱序 / 选项标签替换 / 思考对照 / judge 复核 |
| **调用总量** | **53,379 次**（成功 53,291，失败 88），**37 个权威批次** |
| **口径** | 准确率 = `correct ÷ (correct + wrong)`；`invalid` / `truncated` 不进分母、单独报告 |
| **判分** | 正则解析（严格 / 宽松两口径）+ LLM judge 抽样复核 |

---

## 三、目录结构

```
src/meval/          评测管线本体，按设计分层
  types.py            数据模型
  build_set.py        ① 造题：分层抽样 + 冻结哈希
  adapt.py            ② 请求构造（纯函数 + 哨兵插槽）
  execute.py          ③ 执行：并发 / 退避 / 断点续跑 / 原始落盘
  tally.py            ④ 判分：严格·宽松双口径 + 四档判级
  metrics.py          ⑤ 聚合：四档分账 / McNemar / Kappa / 留一法难度
  client.py           OpenAI 兼容客户端（参数分流 + 代理规避）
  exclusions.py       题集剔除清单（42 题，带指纹）
  paths.py            唯一路径真相来源

scripts/            入口、实验仪器与核校工具（~170 个）
  build_pdf.py        报告 → PDF（pandoc + xelatex + Eisvogel）
  build_pipeline_fig.py  管线图 TikZ → PNG
  ci_gate.py          回归门禁（题集 / 参数 / 口径冻结）
  exp_smoke.py        实验冒烟（124 项）
  _check_external_report.py / _check_polished_report.py   报告一致性自检
  _ratio_pairwise.py  逐对不一致率（§4.3 的主口径）
  _verify_*.py        逐表逐格复算（见溯源文档 §4）

configs/            arms.yaml（五臂 + 各自生效参数）· matrix.yaml（配置矩阵）· exclusions.json
data/               eval_set.jsonl · ceval_set.jsonl · orbench_set.jsonl
                    valid_pool.jsonl · *_relabel_set.jsonl（选项标签替换）
prompts/            c1_strict / c2_free / c5_fewshot / e2_labelshuffle /
                    e6_b_noreq / e6_d_stepwise / judge_cot / judge_refusal
docs/               design.md · limitations.md · open_issues.md · HANDOFF.md
                    param_matrix.md · pipeline.md · exp_runs.md · badcase_method.md
figures/            14 张正式图 + latex/（管线图源码与 HOWTO.md）
runs/               每次跑批：manifest + raw + tallies（已 gitignore）
reports/            对外报告 · 溯源清单 · 批次总表 · 实验报告
tests/              判分与统计单测
```

---

## 四、数据流

```
① 造题 → Instance[]（冻结并哈希）
      │
┌───── 循环 53,379 次 ─────┐
│ ② 构造 prompt → ③ 发请求 → ④ 判分 │
└──────────────────────────┘
      │  raw JSONL（不可变）+ tallies JSONL（可重算）
      ▼
⑤ 聚合指标 → ⑥ 出图 + 竞品报告
```

**核心工程决策**：`raw` 与 `tallies` 分离。改判分规则**不需要重跑 API** —— 窗口期里最可能返工的就是判分逻辑。项目期间判分口径改过多次，一次 API 都没重跑。

---

## 五、快速开始

```powershell
# 1) 环境（不要复制别人的 .venv —— 里面有绝对路径）
uv venv .venv --python 3.12
uv pip install --python ".\.venv\Scripts\python.exe" `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

# 2) 密钥
Copy-Item .env.example .env    # 填入真实 key

# 3) 连通性 + 参数生效探测（★ 必做，结果回填 docs/param_matrix.md）
.\.venv\Scripts\python.exe scripts\smoke_arms.py
.\.venv\Scripts\python.exe scripts\probe_params.py

# 4) 单测 + 两个门禁
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\ci_gate.py      # 当前 PASS 203 / FAIL 1 / WARN 6
.\.venv\Scripts\python.exe scripts\exp_smoke.py    # 当前 124/124

# 5) 出报告（Markdown → PDF）
.\.venv\Scripts\python.exe scripts\build_pdf.py --src reports/report_external_helm_polished.md
```

> `ci_gate.py` 那 1 项 FAIL 是 longcat 的**连接失败**（上游间歇性路由缺陷，非客户端可修），不是口径问题。
> 一键复现演练：`.\run_all.ps1`（小样本走完整条管线，末步即门禁）。

---

## 六、结果速览

> 全部取自落盘数据，复算命令见 [`reports/report_data_provenance.md`](reports/report_data_provenance.md)。
> 跨配置比较一律用**同题配对**（McNemar 精确检验），不得拿两次报告的数字直接相减。

### 6.1 准确率（关闭思考 · 严格口径）

| 模型 | 合并 | MMLU-ProX | C-Eval |
|---|---:|---:|---:|
| `longcat-2.5-preview-free` | **75.8%** | 62.9% | **88.0%** |
| `deepseek-v4.1-flash` | 74.7% | **64.1%** | 84.6% |
| `qwen3.8-flash` | 74.0% | 61.2% | 85.9% |
| `deepseek-v4-pro` | 68.8% | 58.6% | 78.4% |
| `mimo-v2.6-flash` | 63.0% | 49.1% | 75.9% |

**前三名在 1:1 合并集上最大差 1.8pp，低于 5pp 判定阈值，且两题集排序相反** ⇒ 不给出总体排名。
分领域看，三个 Flash 档模型各在 4–5 个领域取得首位，而 `deepseek-v4-pro`（1.6T）与 `mimo-v2.6-flash` 在 14 个领域里**一个都没拿到首位**。

### 6.2 稳定性与位置敏感性（100 题）

| 模型 | 重复 10 次答案全同 | 选项重排翻转率 | 逐对不一致率 重排/重复 → 比值 |
|---|---:|---:|---|
| `longcat-2.5-preview-free` | **87.0%** | 45.2% | 28.3% / 4.6% → **6.19** |
| `deepseek-v4-pro` | 80.6% | 66.7% | 43.0% / 7.9% → **5.47** |
| `deepseek-v4.1-flash` | 77.4% | 50.5% | 33.0% / 8.6% → 3.82 |
| `qwen3.8-flash` | 69.9% | 57.0% | 40.9% / 13.7% → 2.99 |
| `mimo-v2.6-flash` | **28.0%** | 72.0% | 53.8% / 41.0% → **1.31** |

**28.0% 与 87.0% 的跨度远大于单次运行的随机波动** ⇒ 同题重复 10 次答案全同的比例本身就是一条竞品信息。
比值这一列衡量的是「位置改变带来的答案变化，是模型自身抖动的多少倍」：`mimo-v2.6-flash` 最接近 1（1.31 / 1.21），其余四个模型在 **2.0 ~ 11.5 倍**之间。
**本报告不给绝对门槛** —— 比值还受题目难度、作答策略与分母大小影响，能确定的是「`mimo` 的位置敏感度与它自身抖动同量级，其余四个模型显著更高」。

> ★ **比值必须用逐对不一致率，不能用翻转率相除** —— 翻转率（「是否变过」）随观测次数单调上升，
> 选项重排是 4 次观测、重复作答是 10 次，直接相除会把比值系统性压低。

### 6.3 作答格式、示例与思考模式

| 实验 | 结论 |
|---|---|
| **限制格式 vs 自由作答** | 五模型一致上升 **+13.8 ~ +29.1pp**，全部显著 |
| **增益拆解**（A→B→C→D） | 允许写理由 **+15.9**、要求末行答案 **+10.0**、显式要求逐步推理 **−0.4（p=1.00）** |
| **5-shot** | MMLU-ProX **+2.8pp**（p=0.14）、C-Eval **+0.9pp**（p=0.56）—— 示例数量从 1 到 10 不构成变量 |
| **思考开关** | 五模型上升 **+25.0 ~ +36.8pp**（全部显著），且**排名重洗** |

思考开关是**效应最大的单一变量**：关闭时 `deepseek-v4-pro` 落后 `deepseek-v4.1-flash` 10.5pp，
开启后反超 1.4pp 升至首位；另有三个模型开启后准确率**完全相同（86.8%）**。
主表的全部准确率与成本数字都在关闭思考下取得 —— 这是本报告最重要的口径声明。

### 6.4 安全性与成本（OR-Bench 全量 1,974 题 × 4 臂）

| 模型 | 过度拒答率（Hard 1,319） | 危险服从率（Toxic 655） |
|---|---:|---:|
| `qwen3.8-flash` | **82.5%** | **1.4%** |
| `deepseek-v4.1-flash` | 69.8% | 3.2% |
| `longcat-2.5-preview-free` | 54.7% | 3.5% |
| `mimo-v2.6-flash` | **46.6%** | **8.6%** |

**两轴排序完全反转** ⇒ 安全策略上存在取舍，不是单纯的能力优劣。

成本：`deepseek-v4-pro` 输出 30,113 token、中位延迟 2.87s；`qwen3.8-flash` 输出 144,322 token（**4.8 倍**）
而准确率反而更低。**输出 Token 的跨模型差异（4.8 倍）远大于准确率差异（1.2 倍）。**

---

## 七、核心发现

1. **判分口径的价值不是"制造伪低分"，而是"让作答从不可判定变成可判定"。** 作答格式递进实验里，B/C/D 三组在严格口径下都是 **0 条可判**，换成宽松口径后分别恢复到 40.6% / 93.4% / 90.0%。**同一个模型、同一批题，只换口径就从「无法评分」变成「可以评分」。**
2. **强格式约束本身有代价，量级 13.8–29.1pp**（配对显著）。机制不是"自由发挥更好"，而是强约束**拿掉了草稿纸与作答空间**。
3. **参数被忽略 ≠ 输出可复现。** 参数探测阶段实测的**三个臂温度全部被忽略**（同题 ×6 @ `temperature=1.0` 均只产生 1 种输出；`deepseek-v4-pro` 与 `qwen3.8-flash` 未做该探测，五臂的 `temperature=0.1` 是冻结值而非实测结论）。但同题 10 次仍有模型只有 **28.0%** 的题答案全同。
   ⇒ **「固定温度 = 可复现」不成立。** 单次评测自带重复噪声，而这条信息只有重复采样能量化 —— 它衡量的是模型属性，不是抽样误差。
4. **思考开关是效应最大的单一变量**，且会改变排名。任何跨模型对比都必须声明思考状态。
5. **单一总分不足以区分这五个模型。** 前三名差距低于判定阈值、两集排序相反；而分领域、稳定性、安全三个维度上差异显著。
6. **示例的示范作用强于正文指令 —— 一次真实的实验设计缺陷。**（三臂期，C5 首轮）5-shot 首轮只因示例写成 `答案：X`，就造出了「**+11.9 ~ +17.3pp、统计显著**」这个方向明确、统计显著、**且完全错误**的结论；同批数据严格口径有 393/450 条判 `invalid`。把示例改成示范裸字母后，效应塌成 +0.9 / +1.0 / −2.5pp，全部不显著。
   **改了 prompt，第一件事是把渲染结果打印出来人眼过一遍，而不是直接跑 750 次。**

---

## 八、纪律与已知坑

### 不可违反的纪律

1. **`execute` 层不解读模型输出。** 判分逻辑出现在这一层即为 bug。
2. **`tally` 层不发起任何模型调用。** 必须能对同一批 raw 反复重跑。
3. **few-shot 示例只能取自 `valid` split。** 取自 `test` 即数据泄漏。
4. **准确率只能有一个分母。** 分母是 `correct + wrong`，实现只允许有一份：`metrics.tier_accounting()`。
   历史上跑批日志自己算过 `correct / 全部`，于是同一批数据出现两套准确率 —— **两套分母共存比任何一个错值都危险**。
5. **统一题集规则**：截断率高时必须剔除「任一模型没做完」的题，且**在所有模型上剔同一批**。否则被剔掉的难题会让个别模型虚高。

### 踩过的坑（挑几个贵的）

| 坑 | 现象 | 应对 | 固化为 |
|---|---|---|---|
| **空 `content` 静默失败** | 推理模型把预算烧在 reasoning 上，`content` 为空串，**而 HTTP 状态是 200** | `max_tokens≥1024`；读 reasoning；`finish_reason=='length'` 归 `truncated` 而非 `invalid` | `tests/test_truncation_bias.py`（偏倚区间）· `tests/test_tally.py` |
| **judge 思考未关 = 选择性偏差**<br>（三臂期 C4 首轮） | judge 隐藏思考吃光预算，41/450（9.1%）条判不出，**且失败集中在规则判错的题上** → 一致率被高估 | 判分前显式关闭 judge 思考（`reasoning_effort: "none"`），并记录到 manifest | `scripts/_check_judge_verdicts.py`（可复核） |
| **翻转率不能相除** | 两侧观测次数不同（4 次 vs 10 次），比值被系统性低估 | 改用**逐对不一致率**（与观测次数无关） | `tests/test_consistency.py` · `scripts/_ratio_pairwise.py` |
| **C-Eval 选项重排也是乱序的** | 不还原坐标系就跨参照系比较，得到 67–76% 的假值、比值虚高到 12.7–58.4 | 还原到原题坐标系再比（正确值 10.7–30.3%） | `tests/test_position_refusal.py` |
| **示例示范的格式压倒正文指令**<br>（三臂期 C5 首轮） | 示例写 `答案：X` → 模型输出长篇推理 → 严格口径 393/450 判不出，配对差被伪造成 +11.9~+17.3pp | 示例必须示范**目标输出格式本身** | `tests/test_prompt_letters.py` · `adapt.prompt_fingerprint()` 守卫（改模板即拒跑） |
| **厂商私有参数必须走 `extra_body`** | 直传 kwarg 在**本地**抛 `TypeError`，错误信息像网络问题 | 标准参数走 kwargs、私有参数走 `extra_body` | `tests/test_client_params.py` |

---

## 九、产物索引

| 路径 | 内容 |
|---|---|
| [`reports/report_external_helm_polished.md`](reports/report_external_helm_polished.md) | **对外报告**（精简版，25 表 / 14 图 / 22 页 PDF） |
| [`reports/report_external_helm_polished.md`](reports/report_external_helm_polished.md) | **对外报告（唯一权威版）**：6 章 / 26 表 / 14 图 / 24 页 PDF |
| `reports/_archive/report_external_helm_full.md` | 完整版报告（35 表，含全部明细与推导；已冻结归档，仅供技术参考） |
| [`reports/report_data_provenance.md`](reports/report_data_provenance.md) | **数据溯源清单**：35 表 / 14 图 / 52 处含数字正文的来源与复算命令 |
| [`reports/RUN_INVENTORY.md`](reports/RUN_INVENTORY.md) | 37 个权威批次总表 |
| [`reports/polished_report_review.md`](reports/polished_report_review.md) | 精简版逐项核校清单（含已修 / 待修） |
| [`reports/PROJECT_SUMMARY.md`](reports/PROJECT_SUMMARY.md) | 项目全过程总结 |
| `reports/exp_*.md` | 八份补充实验报告（E1/E1b 思考 · E2/E2b 标签替换 · E3 judge · E5 5-shot · E6 递进 · 汇总） |
| [`figures/latex/HOWTO.md`](figures/latex/HOWTO.md) | 管线图怎么改（TikZ 速查 + 四个坑） |
| `figures/30–44_*.png` | 14 张正式图件 |
| [`docs/open_issues.md`](docs/open_issues.md) | 缺陷清单与修复顺序 |
| [`docs/limitations.md`](docs/limitations.md) | 局限必读 |

### 报告的排版

PDF 走 `build_pdf.py`（pandoc + xelatex + Eisvogel），排版参数取自 **CCL 2026 官方中文论文模板**：
宋体正文 + Times New Roman 西文 + 单倍行距 + 首行缩进 1em + 一级标题 12pt 黑体 / 二三级 10pt 黑体。
调研产物（含官方模板原件与编译预览）在 `reports/_verify/web/`。

---

## 十、当前状态

**已完成**

- [x] 五臂连通、参数探测与冻结；题集冻结（MMLU-ProX / C-Eval / OR-Bench）
- [x] 管线四层实现 + 双口径判分 + 四档判级；单测与 `exp_smoke` 124/124
- [x] **37 个权威批次、53,379 次调用**
- [x] 补充实验：思考开关对照（含 longcat 单跑补缺）· 选项标签替换 · judge 复核 · 受控 5-shot · 作答格式递进
- [x] **统一题集**：思考对照统一到 76 题共同集（剔除任一模型没做完的 17 题）
- [x] 判分口径由「翻转率之比」改为「逐对不一致率」
- [x] 对外报告 + 完整报告 + **数据溯源清单**（逐格可复算）+ 核校清单
- [x] 报告排版对齐 CCL 2026 中文模板；管线图重绘
- [x] `ci_gate` PASS 203 / FAIL 1（longcat 连接，非口径问题）· 报告一致性 27/27

**已知局限**

- **badcase 归因停在候选池。** 归因脚本已跑通并产出 100 条复核清单，但 `confirmed_attribution` 字段需人工回填（104 条错题），回填后重跑才产生可用分布。这是上述闭环里唯一没有走完的一环，卡点在人工标注成本。
- `manifest.params_effective` 有 13 处与实际请求参数不符（3 处在权威批次上）。**报告结论不受影响** —— raw 记录才是真值，问题字段不能单独作证据。见 [`reports/RUN_INVENTORY.md`](reports/RUN_INVENTORY.md) §4.6。
- 对外报告仍有若干口径与呈现细节待修，已逐条登记（含 2 条经复核确认**并非**报告错误）。见 [`reports/polished_report_review.md`](reports/polished_report_review.md)。
- 完整局限见 [`docs/limitations.md`](docs/limitations.md)。
