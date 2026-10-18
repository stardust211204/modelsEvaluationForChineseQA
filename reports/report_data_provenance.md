# 报告数据溯源清单

> **★ 适用范围（2026-10-17 补注）**：本清单以 **完整版报告**
> [`_archive/report_external_helm_full.md`](_archive/report_external_helm_full.md) 的
> 表图编号为索引。该版本已于 2026-10-17 **冻结归档**；对外报告已改为精简版
> [`report_external_helm_polished.md`](report_external_helm_polished.md)（6 章 / 27 表 / 14 图）。
>
> 精简版以完整版为主体，同一张表 / 同一张图在两版中的数据来源与算法完全一致，
> 只是**章节编号不同**。因此本清单仍是精简版全部数字的有效复算依据；
> 查表时请按**表的内容**（而非章节号）对应。
>
> **例外（2026-10-18 补）**：精简版后来**新增了若干数字**（`invalid` 率、qwen 的网关拦截拆分、
> 合并口径的权重、剔除题的学科分布、方法学结论收口表），这些在完整版里没有。
> 它们逐条登记在文末 **§六**。

> 逐条记录 [报告](_archive/report_external_helm_full.md) 中**全部 35 张表、14 张图、52 处含数字正文**的
> 数据来源与计算过程。
>
> **命名**：`T` = 表格（按出现顺序编号）、`F` = 图件、`L` = 正文行号。
> **核对方式**：每条给出复算脚本与命令；`§4` 是命令索引。

---

# 一、通用计算规则

报告里反复出现的量，先集中定义一次，后面各条只写「按 §1.x」。

### 1.1 准确率

```
准确率 = correct ÷ (correct + wrong)
```

`invalid`（无解析结果）与 `truncated`（触及长度上限）**不进分母**，单独统计。
分母 = 该批次的计分题数 − 无效/截断题数。

### 1.2 四类判定

| 档位 | 判据 |
|---|---|
| `correct` | 解析出的选项 == 标准答案 |
| `wrong` | 解析出选项但 != 标准答案 |
| `invalid` | 解析不出选项 |
| `truncated` | `finish_reason == "length"` |

### 1.3 严格 / 宽松解析

- **严格**：整段回复必须就是一个裸选项字母（正则 `BARE_RE`）
- **宽松**：在严格的基础上，继续尝试「答案：X」标记 → 选项短语 → 末行（解析阶梯，见 `src/meval/tally.py`）
- **严格 ⊂ 宽松**：宽松包含严格的全部命中

### 1.4 题集剔除

- **固定剔除**：`configs/exclusions.json` 的 42 题（MMLU-ProX 的 LaTeX 公式题 37 + 选项重复 5）。
  对所有模型一致，与批次无关。
- **判分档位剔除**：`invalid` / `truncated` 逐模型发生，因此各模型分母可能差 1–2 题。

### 1.5 配对检验

McNemar 精确检验，**只在两组都可判的条目上做**（`src/meval/metrics.py::mcnemar_exact`）。
报告统一用 **Δ≥5pp 且 p<0.05** 作为「有差异」的判定阈值。

### 1.6 留一法难度

估计某模型的题目难度时，只用**其余四个模型**在该题上的表现，避免被测模型自己影响难度定义。

### 1.7 稳定性 / 选项重排（两个口径，务必区分）

- **重复作答稳定性** = 同题重复 10 次，`answer_index` 完全一致的题目占比
- **重复作答改变率** = 100% − 稳定性
- **选项重排影响（翻转率）** = 同题 4 次观测（原序 1 次 + 乱序 3 次），
  `answer_index` **还原到原题选项坐标系**后，出现过 ≥2 个不同值的题目占比
  - ★ 还原是必须的：乱序轮的作答记在**乱序坐标系**里，不还原会得到 90%+ 的假值
  - ★ `None`（invalid/truncated）**跳过不计**，否则会被当成一个"答案"
- **逐对不一致率** = 任取两次观测，答案不同的概率。与观测次数无关，
  是唯一能跨不同抽样次数比较的口径（`scripts/analyze_position.py` 的主口径）

---

# 二、来源批次总表

**共 37 个权威批次、53,379 次调用**（成功 53,291）。按用途分组。

### 2.1 MMLU-ProX 能力线

| run_id | 配置 | 题数×模型 | 调用 | 用途 | 报告位置 |
|---|---|---:|---:|---|---|
| `c1-5arm-v2` | c1 | 588×5 | 2,940 | **主口径**（限制格式） | T12, F3, F4, F5 |
| `c2hi-5arm-v2` | c2 | 150×5 | 750 | 自由作答 | T20, F7 |
| `c3-5arm-v2` | c3 | 100×10×5 | 5,000 | 重复采样稳定性 | T16, T17 |
| `c4-5arm` | c4 | 150×5 | 750 | judge 复核（思考**未**关，对照） | — |
| `c4-5arm-nothink` | c4 | 150×5 | 750 | **judge 复核（思考关）** | T7 |
| `c5-5arm-v3` | c5 | 150×5 | 750 | 5-shot 初版 | T24 |
| `c6-5arm-v2` | c6 | 100×4×5 | 2,000 | 选项乱序 | T16, T17, T18 |
| `e2-relabel-5arm` | e2 | 100×5 | 500 | 选项标签替换（乱排版+破对应） | T19 |
| `e2b-relabel-5arm` | e2b | 100×5 | 500 | 选项标签替换（仅乱排版） | T19 |
| `e6-a-5arm` | c1 | 150×5 | 750 | 作答格式递进 A（只答字母） | T22, F8 |
| `e6-b-5arm` | e6b | 150×5 | 750 | 递进 B（可写理由，不要求末行） | T22, F8 |
| `e6-c-5arm` | c2 | 150×5 | 750 | 递进 C（要求末行答案） | T22, F8 |
| `e6-d-5arm` | e6d | 150×5 | 750 | 递进 D（再要求逐步推理） | T22, F8 |
| `e1-think-none-5arm` | c3 | 100×5 | 500 | **思考对照·关闭** | T28, F12 |
| `e1-think-high1024-5arm` | c3 | 100×5 | 512 | 思考对照·开启（1024 预算） | — |
| `e1-think-high8192-5arm` | c3 | 100×5 | 518 | **思考对照·开启（8192 预算）** | T28, F12 |
| `e1b-longcat-think1024` | c3 | 100×1 | 100 | longcat 思考开启（1024） | — |
| `e1b-longcat-think8192` | c3 | 100×1 | 128 | **longcat 思考开启（8192，可比口径）** | T28, F12 |
| `e5-mmlu-n{1,3,5,10}-5arm` | c5 | 150×5 ×4 | 3,000 | 5-shot 梯度 | T23, F9 |

### 2.2 C-Eval 能力线

| run_id | 配置 | 题数×模型 | 调用 | 用途 | 报告位置 |
|---|---|---:|---:|---|---|
| `ceval-c1-5arm-v2` | c1 | 583×5 | 2,915 | **主口径** | T12, F3, F4, F6 |
| `ceval-c2hi-5arm-v2` | c2 | 150×5 | 750 | 自由作答 | T20, F7 |
| `ceval-c3-5arm` | c3 | 100×10×5 | 5,000 | 重复采样稳定性 | T16, T18 |
| `ceval-c5-5arm-v3` | c5 | 150×5 | 750 | 5-shot 初版 | T24 |
| `ceval-c5b-5arm` | c5b | 150×5 | 750 | 5-shot 选项数匹配（初版对照） | T24 |
| `ceval-c6-5arm` | c6 | 100×4×5 | 2,000 | 选项乱序 | T16, T18 |
| `e5-ceval-n0-5arm` | c2 | 150×5 | 750 | 5-shot 梯度 n=0 基线 | T23, F9 |
| `e5-ceval-n{1,3,5,10}-5arm` | c5 | 150×5 ×4 | 3,000 | 5-shot 梯度 | T23, F9 |

### 2.3 安全线

| run_id | 题集 | 调用 | 用途 | 报告位置 |
|---|---|---:|---|---|
| `orbench-full` | or-bench | 5,922 | 3 个模型源响应 | F10 |
| `orbench-qwen` | or-bench | 1,974 | qwen 源响应（26 条失败） | F10 |
| `orbench-judge-full` | refusal_set | 5,922 | **judge 判定** | T25, F10 |
| `orbench-judge-qwen` | refusal_set | 1,948 | judge 判定（qwen） | T25, F10 |

### 2.4 限额模型与辅助

| run_id | 调用 | 用途 | 报告位置 |
|---|---:|---|---|
| `sb-c1-eval` | 588 | 限额模型画像 | T29 |
| `sb-c1-ceval` | 583 | 限额模型画像 | T29 |
| `endpoint-cmp` | 300 | 端点一致性（辅助探针） | 正文提及 |

### 2.5 跨章节的批次

| 批次 | 覆盖报告的节 |
|---|---|
| `c1-5arm-v2` | §3.2 准确率、§3.3 学科、§3.3 难度、§3.5 指令遵循、§5.1 配比 |
| `ceval-c1-5arm-v2` | 同上（C-Eval 侧） |
| `c3-5arm-v2` / `ceval-c3-5arm` | §3.4 稳定性 |
| `c6-5arm-v2` / `ceval-c6-5arm` | §3.5 选项顺序 |
| `e1-think-none-5arm` + `e1-think-high8192-5arm` + `e1b-longcat-think8192` | §3.9 思考开关、§3.2/§3.9 口径声明、§5 局限 |

---

# 三、逐节清单

## §1.1 评测对象

### T1 模型规格表（行 13）
| 列 | 来源 | 说明 |
|---|---|---|
| 创建者 / 模型 | 人工整理 | 五个被测模型 |
| 总参数 / 激活参数 / 上下文 / 模态 | **厂商公开资料** | 非本项目测量；`longcat-2.5` 厂商未公开 |

**⇒ 无需复算。** 这是报告里唯一不来自 `runs/` 的表。

## §1.2 评测范围与规模

### T2 规模表（行 28）
| 行 | 值 | 来源 | 计算方式 |
|---|---|---|---|
| 数据来源 | MMLU-ProX / C-Eval / OR-Bench | — | 固定 |
| Benchmark | MMLU-ProX 全量 + C-Eval 抽样、OR-Bench 全量 | `data/*.jsonl` | 见 §2.2 各题集行 |
| **模型调用** | **53,379（成功 53,291）** | 37 个权威批次 | `raw/` 下 jsonl 记录条数之和 |
| 评测设置 | 9 项 | — | 对应 §2.6 的 T8 |

**复算**：`python scripts/_count_authoritative.py`
**核对点**：该脚本对原 15 个批次加总得 39,371，与 `RUN_INVENTORY.md` 一致。

### L35 限额模型说明（正文）
「思考默认开启且无法关闭」——**证据来自 T29 的两个批次**：`sb-c1-eval` 588 条中 91% 有推理内容、
`sb-c1-ceval` 583 条中 75%，且生效参数均为 `chat_template_kwargs: {enable_thinking: false}`。

**复算**：读 `runs/sb-*/raw/` 的 `reasoning` 字段长度 + `runs/sb-*/manifest.json` 的 `params_effective`。

---

## §2.1 评测管线与数据流

### F1 `40_pipeline_main.png` / F2 `41_pipeline_safety.png`
**无数据**，是管线定义图。
- 源文件：`figures/latex/pipeline_main_v11.tex` / `figures/latex/pipeline_safety_v13.tex`
- 编译：`xelatex <file>.tex` → `pdftoppm -png -r 150 <file>.pdf` → 复制到 `figures/`
- 图上的「思考关闭」标注对应 §3.9 的结论

### L41 管线叙述（正文）
「判分阶段引入 LLM judge：在准确率评测管线用它对规则判分做抽样复核」
⇒ 对应 T7（一致率）与 `c4-*` 批次。

---

## §2.2 Benchmark

### T3 题集表（行 51）
| 列 | 来源 | 计算方式 |
|---|---|---|
| MMLU-ProX Benchmark 构建 588 | `data/eval_set.jsonl` | 冻结题集的行数 |
| MMLU-ProX 记分题目 546 | 588 − 42 | 剔除 `configs/exclusions.json`（37 LaTeX + 5 重复选项）|
| C-Eval Benchmark 构建 583 / 记分 583 | `data/ceval_set.jsonl` | 冻结题集行数；**无剔除** |
| OR-Bench 1,974 / 1,974 | `data/orbench_set.jsonl` | Hard 1,319 + Toxic 655 |
| 来源 / 结构 | 论文与题集文档 | 非本项目测量 |

**复算**：`python scripts/build_set.py`（重建）、`python scripts/build_exclusions.py`（剔除清单）
**注意**：`exclusions.json` 的 `rule_fingerprint` = `23c9c5c3e131`，改题集会使该指纹变化。

### L57–L67 三个题集的背景叙述
均为**论文转述**（[2503.10497] / [2305.08322] / [2405.20947]），不含本项目数据。
唯一例外：L57 末尾「已滤除 42 道题」⇒ 同 T3。

---

## §2.3 Prompt生成

### T4 作答格式递进表（行 94）
**无数据**，是四档指令的差异对照。
- 实物：`prompts/c1_strict.txt`(A) / `e6_b_noreq.txt`(B) / `c2_free.txt`(C) / `e6_d_stepwise.txt`(D)
- **核对方式**：直接 diff 四个文件

### L71–L104 各任务模板
| 任务 | prompt 文件 |
|---|---|
| 直接作答 | `prompts/c1_strict.txt` |
| 自由作答 | `prompts/c2_free.txt` |
| 递进 B / D | `prompts/e6_b_noreq.txt` / `e6_d_stepwise.txt` |
| 5-shot | `prompts/c5_fewshot.txt`（`{demos}` 槽位） |
| 选项标签替换 | `prompts/e2_labelshuffle.txt` |

**注意**：`{last}` / `{question}` / `{options}` / `{demos}` 是模板槽位；
`{demos}` 用**哨兵插入**（`src/meval/adapt.py::render_prompt`），避免示例正文里的花括号触发 `str.format` 崩溃。

---

## §2.4 判定方法

### T5 四类判定表（行 111）
**定义性内容**，对应 `src/meval/types.py::Tier` 与 `src/meval/tally.py` 的解析阶梯。
**复算**：`python scripts/audit_parse_ladder.py`

### T6 两口径可判条目表（行 124）
| 列 | 来源批次 | 计算方式 |
|---|---|---|
| 只答一个字母 | `e6-a-5arm` | 严格 / 宽松口径下 `correct+wrong` 的条目数 |
| 可写理由，不要求末行 | `e6-b-5arm` | 同上 |
| 可写理由，要求末行 | `e6-c-5arm` | 同上 |
| 再加逐步推理 | `e6-d-5arm` | 同上 |

- 题集：MMLU-ProX，同一批 150 题 × 5 模型 = 680 条
- **复算**：`python scripts/exp_e6_analysis.py`
- **核对点**：严格口径在三组自由作答下均为 **0** —— 因为严格解析要求整段回复是裸字母

### T7 judge 复核一致率表（行 135）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 覆盖率 | `c4-5arm-nothink` | judge 判出 correct/wrong 的条目 ÷ 150 |
| 一致率 | 同上 + `c1-5arm-v2` | judge 判定 vs **规则**判定的相同比例 |
| Cohen κ | 同上 | 二元一致性系数，只在两侧均判出 correct/wrong 的条目上算 |

- **落盘文件**：`runs/c4-5arm-nothink/judge_vs_rule.json` ← **T7 的直接来源，可直接读**
- **计算函数**：`scripts/run_judge.py::judge_vs_rule()`（写 `report["arms"][arm]["binary"]`）
- **judge 指令**：`prompts/judge_cot.txt`
- **生成命令**（已执行过，重跑会覆盖）：
  ```bash
  python scripts/run_judge.py --source-run c1-5arm-v2 --source-config c1 \
    --run-id c4-5arm-nothink --arms <5 个模型> --judge-param reasoning_effort=none
  ```
- ★ **两个易错点**：
  1. **不要用 `c4-5arm`**（judge 思考未关）—— 它的覆盖率只有 85–89%，
     一致率虽仍显示 100% 但样本被削掉了一截。必须用 `c4-5arm-nothink`。
  2. **不要用 `scripts/report_metrics.py`** —— 它的 `judge_agreement_report()`
     算的是**严格 vs 宽松**两个解析口径之间的一致性，与 judge-vs-规则无关，名字有误导性。

---

## §2.6 评测任务

### T8 任务设置表（行 157）
「选取题目数」列全部来自各批次的 manifest；「用途」是设计意图。对应关系见 §二 的来源批次总表。

---

## §2.7 安全评测

### T9 子集表（行 173）
| 行 | 来源 |
|---|---|
| Hard 1,319 | `data/orbench_set.jsonl` 中 `subset=hard` |
| Toxic 655 | 同上 `subset=toxic` |

**复算**：`python scripts/build_orbench_set.py`

### T10 judge 判据表（行 182）
**定义性内容**，逐字来自 `prompts/judge_refusal.txt`。
**核对方式**：直接读该文件。

### T11 指标分母分子表（行 193）
| 指标 | 计算方式 |
|---|---|
| 过度拒答率 | Hard 侧被判 `REFUSE` 的条目 ÷ 1,319 |
| 危险服从率 | Toxic 侧被判 `COMPLY` 的条目 ÷ 655 |

**复算**：`python scripts/run_refusal_judge.py` + `src/meval/metrics.py::refusal_metrics`

### L198 / L457 部分服从为 0（正文）
- 来源：`orbench-judge-full`（5,922）+ `orbench-judge-qwen`（1,948）= **7,870 条判定**
- 计算方式：统计 judge 输出里 `PARTIAL` 标记的出现次数
- **复算**：`python scripts/_check_judge_verdicts.py`
- ★ **易错点**：分母是 **7,870**（judge 实际判定数），不是 7,896（源响应数）。
  qwen 侧有 26 条源响应请求失败、未送判。

### L200 judge 思考状态（正文）
- 来源：`runs/orbench-judge-full/manifest.json` 与 `runs/orbench-judge-qwen/manifest.json`
- 证据：`params_effective` 含 `reasoning_effort: "none"`；`raw/` 里 `reasoning` 非空占比 0%
- 「12% 判不出」来自 `c4-5arm`（思考未关）的覆盖率 85–89%

---

## 主要发现（不编号）

### L206–L211 六条发现（正文）
全部是**其他章节结论的转述**，无独立数据。逐条对应：

| 条 | 数据来源 |
|---|---|
| 1 准确率 3.4% | T12 / T33（§5.1） |
| 2 稳定性 28–87% | T16 |
| 3 指令遵循 13.8–29.1% | T20；拆解 +15.9/+10.0/−0.4 来自 T22 |
| 4 5-shot +2.8/+0.9 | T23 |
| 5 安全性排序反转 | T25 |
| 6 思考开关 25.0–36.8 | T28 |

---

## §3.1 准确率

### F3 `30_overall_barh.png`（行 219）
- 来源：`c1-5arm-v2`、`ceval-c1-5arm-v2`
- 生成：`python scripts/make_figures_v3.py --only 1`
- 口径：**严格解析、关闭思考**

### T12 准确率表（行 221）
| 列 | 计算方式 |
|---|---|
| MMLU-ProX | `c1-5arm-v2`，按 §1.1 |
| C-Eval | `ceval-c1-5arm-v2`，按 §1.1 |
| 合并 | 两集 `correct` 之和 ÷ 两集 `(correct+wrong)` 之和（**按题目数加权，不是两列平均**）|

**复算**：`python scripts/competitive_analysis.py`
**核对点**：合并列必须是加权，不是算术平均 —— 两集题量不同（546 vs 583）。

### L215 重复运行标准差 0.6%–1.0%（正文）
- 来源：`c3-5arm-v2`（10 次重复）
- 计算方式：各模型 10 次运行的准确率标准差；折算全量 = 标准差 × √(546/93)
- **复算**：`python scripts/analyze_stability.py`

### L217 口径声明（正文）
数据来自 T28 的思考对照。

---

## §3.2 学科影响

### F4 `31_domain_heatmap.png`（行 233）
- 来源：`c1-5arm-v2`、`ceval-c1-5arm-v2`
- 生成：`python scripts/make_figures_v3.py --only 2`
- 领域映射：`scripts/ceval_domain_map.py`（C-Eval 52 学科 → MMLU-ProX 14 领域）

### T13 各模型取得首位的领域（行 235）
- 来源：同上
- 计算方式：逐领域取 `correct ÷ (correct+wrong)` 最高的模型，统计其取得首位的领域数
- **复算**：`python scripts/merged_domain_conclusions.py`

### L243 首位与末位差距 11%–26%（正文）
- 计算方式：逐领域 `max(准确率) − min(准确率)`
- 注意：这是**极差**，不涉及显著性检验

---

## §3.3 难度影响

### F5 `32_difficulty_spread.png`（行 247）
- 来源：`c1-5arm-v2`
- 生成：`python scripts/make_figures_v3.py --only 3`
- 难度定义：**留一法**（§1.6）

### T14 难度分箱表（行 251）
| 列 | 计算方式 |
|---|---|
| 难度 | 留一法难度分 5 箱（≈0% / 25% / 50% / 75% / ≈100%）|
| 模型间最大差距 | 每箱内 `max(准确率) − min(准确率)` |

**复算**：`python scripts/difficulty_loo.py`

### F6 `36_ceval_levels.png`（行 263）+ T15 层级表（行 265）
- 来源：`ceval-c1-5arm-v2`
- 层级来自 **C-Eval 自带标签**（初中/高中/大学/职业专业），非留一法
- 生成：`python scripts/make_figures_v3.py --only 7`
- **复算**：`python scripts/difficulty_proxy.py`

---

## §3.4 稳定性

### T16 稳定性与选项重排表（行 281）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 重复作答稳定性 | `c3-5arm-v2` / `ceval-c3-5arm` | 10 次 `answer_index` 完全一致的题目占比（§1.7）|
| 选项重排影响 | `c6-5arm-v2` / `ceval-c6-5arm` | 翻转率（4 次观测，**还原坐标系、跳过 None**，§1.7）|

**复算**：`python scripts/analyze_stability.py`、`python scripts/analyze_position.py`
★ **易错点（三处，都会算出完全不同的数）**：
1. 比较 `tier` 而不是 `answer_index` ⇒ 两个不同的错选项被当成"没变"
2. 不把乱序轮**还原到原题坐标系** ⇒ 得到 90%+ 的假值
3. 不跳过 `None` ⇒ 多算 2–3 题（longcat 会从 45.2% 变成 48.4%）

### T17 比值表 MMLU-ProX（行 299）+ T18 比值表 C-Eval（行 309）
**口径 = 逐对不一致率**（2026-10-13 由翻转率改为此口径）。

| 列 | 计算方式 |
|---|---|
| 选项顺序逐对不一致率 | 选项重排的**任意两次观测**答案不同的概率；答案先还原到原题坐标系、跳过 `None` |
| 重复作答逐对不一致率 | 重复采样的**任意两次观测**答案不同的概率 |
| 比值 | 前者 ÷ 后者 |

**复算**：`python scripts/_ratio_pairwise.py`
**★ 为什么不能用翻转率**：翻转率（「是否变过」）随观测次数单调上升。
选项重排 4 次、重复作答 10 次 ⇒ 分母被抬高、比值被系统性低估。
逐对不一致率与观测次数无关，两侧天然可比。

**★ 两个极易踩的坑**：
1. **C-Eval 的选项重排同样是乱序的**，`permuted` 必须传 `True`。
   传 `False` 会跨越坐标系比较，得到 67–76.5% 的假值、比值虚高到 12.7–58.4
   （正确值 10.7–30.3%，比值 2.02–11.47）。
2. `analyze_position.py` 在 **C-Eval 上会输出 `nan`** —— 它把题集池硬编码为
   `eval_set.jsonl`（MMLU-ProX），C-Eval 的题号全部匹配不上。
   所以 C-Eval 只能用 `_ratio_pairwise.py` 复算。

**交叉验证**：`_ratio_pairwise.py` 的 MMLU-ProX 结果与
`analyze_position.py` 的主口径精确一致（43.0 / 28.3 / 33.0 / 40.9 / 53.8）；
C-Eval 结果与本仓另一独立实现一致。

### T19 选项标签替换表（行 332）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 直接作答 | `c1-5arm-v2` | §1.1，**限到 E2b / E2 的同一批 100 题（剔 7 题后 93 题）** |
| 仅乱排版 | `e2b-relabel-5arm` | §1.1 |
| 乱排版 + 破对应 | `e2-relabel-5arm` | §1.1 |
| 排版效应 / 对应效应 / 合计 | 三列相减 | 见下 |

- **排版效应** = 仅乱排版 − 直接作答
- **对应效应** = 乱排版+破对应 − 仅乱排版
- **合计** = 乱排版+破对应 − 直接作答
- 显著性：`metrics.mcnemar_exact`，在**两组都可判的条目**上做
- **复算**：`python scripts/exp_e2b_analysis.py`
- **题集构造**：`python scripts/build_e2b_relabel_set.py`（关键约束 `L[gold] = gold`）
- ★ **易错点（2026-10-18 修）**：C1 是 **546 题**的全量批次，E2b / E2 只在同一批 93 题上跑过。
  直接拿 C1 的全量准确率当基线，等于**用两个不同题集的差当「效应」** ——
  实测会把排版效应从 −2.2 抬到 −7.1、五模型均值从 −2.6 抬到 −8.5，合计从 −5.7 抬到 −11.6。
  所以 `scripts/exp_e2b_analysis.py` 的 C1 分支必须用 `limitfile` 限题（该脚本 `RUNS` 的第四项）。
  两侧的 42 题源缺陷剔除清单原本就已正确应用，问题只在题集范围。

---

## §3.5 指令遵循

### F7 `38_format_constraint.png`（行 349）
- 来源：`c1-5arm-v2`、`c2hi-5arm-v2`、`ceval-c1-5arm-v2`、`ceval-c2hi-5arm-v2`
- 生成：`python scripts/make_figure_format_constraint.py`

### T20 限制格式 vs 自由作答（行 353）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 限制格式作答 | `c1-5arm-v2` | **严格**口径（§1.1）|
| 自由作答 | `c2hi-5arm-v2` | **宽松**口径（§1.1）|
| 提升 | 两列相减 | |
| p 值 | 同上 | McNemar，配对子集 |

**复算**：`python scripts/c2_vs_c1_analysis.py`
★ **注意口径不同**：限制格式用严格、自由作答用宽松 —— 因为自由作答下严格口径命中为 0（T6）。

### T21 分领域净变化表（行 371，14 行）
| 列 | 计算方式 |
|---|---|
| 题数 | 该领域在 150 题里的题目数 |
| 净变化 | 「由错变对」−「由对变错」的题目数 |
| p | McNemar 精确检验 |
| 判定 | 按各领域实际分歧数算出的最小可检出净变化 |

**复算**：`python scripts/c2_vs_c1_analysis.py`
**注意**：「不显著」不等于「没有影响」—— 门槛随领域题量变化（6–12 例）。

### T22 作答格式递进拆解表（行 396）
| 列 | 来源 | 计算方式 |
|---|---|---|
| A→B / B→C / C→D / A→C | `e6-{a,b,c,d}-5arm` | 两组的**共同可判子集**上的准确率差 |
| p | 同上 | McNemar（b/c 计数）|
| 配对题数 | 同上 | 两组都可判的条目数 |

**复算**：`python scripts/exp_e6_paired.py`
★ **为什么必须用配对子集**：四组的可判条目数差别很大（680 / 276 / 635 / 612），
直接用各自分母算出的百分比不可相减。

### F8 `43_format_decomposition.png`（行 405）
- 来源：同 T22
- 生成：`python scripts/make_figures_supplementary.py`
- ★ **口径与 T22 不同**：图用「每模型在自己的可判集上」的阶梯，
  数值为 +16.5 / +10.4 / +0.5，与表的 +15.9 / +10.0 / −0.4 略有差异。
  原因：E6 的 B 组里**不存在五个模型共同可判的题**（0 道），无法构造统一题集。

### L363 / L365 合并净变化（正文）
- MMLU-ProX：173 题由错变对、30 题由对变错，净 +143，p<1e-24
- C-Eval：净 +39，p=2e-05
- **复算**：同上

### L367 提升不是判分放宽造成的（正文）
- 「限制格式作答时 99.9% 是裸字母」⇒ 统计 `c1-5arm-v2` 的 `tier_step == "bare"` 占比
- 「自由作答下严格口径命中率为 0」⇒ 同 T6
- **复算**：`python scripts/audit_parse_ladder.py`

---

## §3.6 5-shot

### T23 两评测集增益表（行 421）
| 列 | 来源 | 计算方式 |
|---|---|---|
| MMLU-ProX +2.8 | `e6-a-5arm`(n=0) vs `e5-mmlu-n10-5arm` | 各模型准确率差的均值 |
| C-Eval +0.9 | `e5-ceval-n0-5arm` vs `e5-ceval-n10-5arm` | 同上 |
| p | 同上 | McNemar |

**复算**：`python scripts/exp_e5_analysis.py`
★ **易错点**：0-shot 基线必须**同期新跑**。
用 `e6-a-5arm`（MMLU-ProX）与 `e5-ceval-n0-5arm`（C-Eval），
不能拿 `c1-5arm-v2` 顶替 —— 后者是 eval_set、不含 C-Eval。

### F9 `44_shot_gradient.png`（行 428）
- 来源：`e5-{mmlu,ceval}-n{0,1,3,5,10}-5arm`
- 生成：`python scripts/make_figures_supplementary.py`
- **注意**：曲线用的是各档**自己的可判分母**（未统一题集），
  `mimo-v2.6-flash` 在个别档少 4–12 题。

### T24 受控 5-shot 逐模型对照（行 432）
| 列 | 来源 |
|---|---|
| 0-shot | `e5-ceval-n0-5arm` |
| 原设置 5-shot | `ceval-c5-5arm-v3`（示例共用，字母范围冲突）|
| 受控 5-shot | `e5-ceval-n5-5arm`（选项数匹配）|

- **复算**：`python scripts/c5_ceval_attribution.py`
- 示例选项数：原设置均值 8.4、目标题 4 ⇒ 冲突来源

---

## §3.7 安全性

### F10 `34_refusal_twosided.png`（行 446）
- 来源：`orbench-judge-full`、`orbench-judge-qwen`
- 生成：`python scripts/make_figures_v3.py --only 5`

### T25 安全指标表（行 448）
| 列 | 计算方式 |
|---|---|
| 过度拒答率 | Hard 侧判 `REFUSE` ÷ Hard 总数 |
| 危险服从率 | Toxic 侧判 `COMPLY` ÷ Toxic 总数 |
| 部分服从 | 判 `PARTIAL` 的比例（实测为 0）|

- **复算**：`python scripts/report_metrics.py`（或 `run_refusal_judge.py` 的汇总输出）
- **judge 指令**：`prompts/judge_refusal.txt`
- **注意**：判分方是 `deepseek-v4-pro`，本身也是被测模型 ⇒ 所以按模型分列以暴露同厂偏好

---

## §3.8 成本效率

### F11 `33_cost_scatter.png`（行 461）
- 来源：`c2hi-5arm-v2`
- 生成：`python scripts/make_figures_v3.py --only 4`

### T26 成本表（行 467）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 准确率 | `c2hi-5arm-v2` | 宽松口径（§1.1）|
| 输出 token | 同上 | `raw/` 的 `tokens_out` **求和**（150 题合计）|
| 中位延迟 | 同上 | `raw/` 的 `latency_s` **中位数** |

**复算**：`python scripts/competitive_analysis.py`（行 99 `sum(...)`、行 105 中位数）
★ **易错点**：token 列是**合计**不是均值。实测五个模型合计为
30,113 / 39,682 / 64,439 / 89,953 / 144,322；对应的**均值**是
200.8 / 264.5 / 429.6 / 599.7 / 962.1 ——
后者正是正文 L341「平均输出为 200 至 962 token」的来源，两处不要混。

---

## §3.9 同厂旗舰档定位

### T27 同厂对照表（行 481）
| 行 | 来源 |
|---|---|
| 基线 / MMLU-ProX | `c1-5arm-v2` |
| 自由作答 / MMLU-ProX | `c2hi-5arm-v2` |
| 5-shot / MMLU-ProX | `e5-mmlu-n5-5arm`（受控 n=5，示例选项数与目标题匹配）|
| 基线 / C-Eval | `ceval-c1-5arm-v2` |
| 自由作答 / C-Eval | `ceval-c2hi-5arm-v2` |

**复算**：`python scripts/competitive_analysis.py`

### T28 思考对照表（行 495）
| 列 | 来源 | 计算方式 |
|---|---|---|
| 关闭思考 | `e1-think-none-5arm` | 严格口径，**76 题共同集** |
| 开启思考 | `e1-think-high8192-5arm` + `e1b-longcat-think8192` | 同上 |
| 效应 | 两列相减 | |
| p | 同上 | McNemar 精确检验 |

- **复算**：`python scripts/exp_e1_same_set.py`
- ★★ **易错点（本节最容易出错的地方）**：
  1. **必须用 76 题共同集**（剔除任一模型在 8192 下没做完的 17 题）。
     用逐模型 93 题会得到 57.0 / 50.5 等值，与表里的 61.8 / 51.3 不一致。
  2. longcat 的开启值必须取 `e1b-longcat-think8192`（1 个模型单独跑），
     不能取 `e1-think-high8192-5arm`（那里 longcat 的 5 道长推理题连接失败）。
  3. **不要用 `e1-think-high1024-5arm`** —— 它 31% 截断，准确率被高估 7.8pp。

### F12 `42_thinking_switch.png`（行 505）
- 来源：同 T28
- 生成：`python scripts/make_figures_supplementary.py`

### L503 排序变化（正文）
- 「由落后 6.5 变成领先 1.4」**这句现在用的是混合口径**：
  - 6.5 = 93 题逐模型口径（flash 57.0 vs pro 50.5）
  - 1.4 = 76 题共同集口径（88.2 vs 86.8）
- **与同节表格一致应改为 10.5**（61.8 − 51.3）
- **复算**：`python scripts/exp_e1_same_set.py` 的 §③ 对照输出

---

## §4 限额模型

### T29 限额模型画像表（行 517）
| 列 | 来源 | 计算方式 |
|---|---|---|
| MMLU-ProX 准确率区间 | `sb-c1-eval` | 区间来自不同 run 的波动 |
| C-Eval 准确率 | `sb-c1-ceval` | — |
| 截断率 | 同上 | `finish_reason == "length"` 占比 |

**复算**：读 `runs/sb-*/raw/`
**注意**：该模型**已下线**，无法重跑；数据是窗口期内采集的。

### L515 / L522 正文数字
- 「1,171 次调用中 75%–91% 有推理内容」⇒ 两个 sb 批次合计
- 「约 14% 请求无输出」⇒ 截断率
- 「截断题在其余模型上 35.4%、未截断题 60.4%」⇒ 留一法难度（§1.6）

---

## §5.1 结论的边界

### F13 `35_cross_dataset.png`（行 532）+ F14 `37_mix_sensitivity.png`（行 536）
- 来源：`c1-5arm-v2`、`ceval-c1-5arm-v2`
- 生成：`python scripts/make_figures_v3.py --only 6,8`

### L534 配对差值（正文）
- 计算方式：两评测集上各模型准确率的配对差；方向翻转的配对落在 ±5% 内
- **复算**：`python scripts/competitive_analysis.py`

### L538 配比敏感性（正文）
- 计算方式：把两集按不同权重合并（C-Eval 占比 50% → 20%），看首位是否变化
- **复算**：`python scripts/make_figures_v3.py`（图 37 的计算部分）

### L544 时段漂移（正文）
| 值 | 来源 | 计算方式 |
|---|---|---|
| −0.1 ~ −6.5（均值 −2.2）| 新跑 vs `c3-5arm-v2` | 同题同参数、仅时段不同的两组对照 |
| −4.7 ~ +2.0（均值 −1.6）| `ceval-c5b-5arm` vs `e5-ceval-n5-5arm` | 逐条同示例、仅时段不同 |

**复算**：`python scripts/exp_e1_analysis.py`（第一组）、`python scripts/exp_e5_analysis.py`（第二组）

### T30 其余结论边界表（行 548）
| 行 | 数据 | 来源 |
|---|---|---|
| 1 | 安全维度仅覆盖 4 个模型 | `orbench-*` 只有 3+1 个模型；`deepseek-v4-pro` 是判卷方 |
| 2 | C-Eval 单学科题目数 5–24 | `data/ceval_set.jsonl` 按学科计数 |
| 3 | 选项顺序不变性排序两集不一致 | T16 第二列的 MMLU-ProX 与 C-Eval 排序对照 |

**行 3 的复算**：`python scripts/analyze_position.py --c6-run c6-5arm-v2 --c3-run c3-5arm-v2`
（MMLU-ProX）与 `--c6-run ceval-c6-5arm --c3-run ceval-c3-5arm`（C-Eval），比较两次输出的模型序。

---

## §5.2 数据与判分的边界

### T31 表（行 558）
| 行 | 数据 | 来源 |
|---|---|---|
| 1 | 42 题剔除、计分 546 | `configs/exclusions.json` |
| 2 | 截断偏差 28.1pp（65.4% vs 93.5%）| `e1b-longcat-think1024` 与 `e1b-longcat-think8192` 的对照 |
| 3 | 34 条弱标记、5 条疑似有误、最大 1.2% | `scripts/audit_parse_ladder.py` |
| 4 | 分母 546/546/544/544/546 与 583/583/582/582/583 | `c1-5arm-v2` / `ceval-c1-5arm-v2` 逐模型统计 |

**行 2 的复算**：取 1024 组里被截断的 31 题，看它们在 8192 组的结果 ⇒ 65.4%；
未截断的 62 题 ⇒ 93.5%。差 28.1pp。

---

## §5.3 设计与覆盖的边界

### T32 表（行 569）
第 3、4、5 项引用 §3.9 / §3.7 / §3.7 的结论，无独立数据。

### L577 受控 5-shot 收敛（正文）
同 T24。

---

## §6.1 图表索引

### T33（行 585，14 行）
**元数据表**，逐图列出出现位置与来源 run_id。本溯源文档 §三 已逐条展开。

---

## §6.3 术语 / §6.4 依据来源

### T34（行 610）/ T35（行 628）
**定义性与文献性内容**，无本项目数据。
T35 的「依据来源」列指向题集论文与官方文档。

---

# 四、复算命令索引

按依赖顺序排列。全部从仓库根目录执行。

### 4.1 环境与冻结件校验

```bash
python scripts/ci_gate.py          # 门禁：题集/参数/指纹冻结检查
python scripts/exp_smoke.py        # 冒烟：124 项
python scripts/_check_external_report.py   # 报告一致性：27 项
```

### 4.2 批次与规模

```bash
python scripts/_count_authoritative.py   # T2 的 53,379
python scripts/_count_calls.py           # runs/ 全量统计
python scripts/run_audit.py              # 批次审计
```

### 4.3 准确率 / 学科 / 难度 / 成本（§3.1–§3.3、§3.8）

```bash
python scripts/competitive_analysis.py   # T12, T26, T27, L534
python scripts/difficulty_loo.py         # T14
python scripts/difficulty_proxy.py       # T15
python scripts/merged_domain_conclusions.py  # T13
```

### 4.4 稳定性与选项顺序（§3.4）

```bash
python scripts/analyze_stability.py                                   # T16 第一列
python scripts/analyze_position.py --c6-run c6-5arm-v2 --c3-run c3-5arm-v2   # T16-T18
python scripts/exp_e2b_analysis.py                                    # T19
```

### 4.5 指令遵循与 5-shot（§3.5–§3.6）

```bash
python scripts/c2_vs_c1_analysis.py      # T20, T21
python scripts/exp_e6_paired.py          # T22
python scripts/exp_e5_analysis.py        # T23, T24, L544
python scripts/c5_ceval_attribution.py   # T24 的错配归因
```

### 4.6 安全（§3.7）

```bash
python scripts/run_refusal_judge.py      # judge 判定
python scripts/_check_judge_verdicts.py  # L198/L457 的 7,870 与 PARTIAL=0
python scripts/report_metrics.py         # T25
```

### 4.7 思考开关（§3.9）

```bash
python scripts/exp_e1_same_set.py        # T28 的 76 题共同集口径 ★
python scripts/exp_e1_unified8192.py     # 统一剔除集口径
python scripts/exp_e1b_analysis.py       # longcat 单模型
```

### 4.8 判定方法（§2.4）

```bash
python scripts/audit_parse_ladder.py     # T5, T6, L367
# T7 的直接来源是落盘文件，无需重算：
#   runs/c4-5arm-nothink/judge_vs_rule.json
# 重算（会覆盖该文件）：
python scripts/run_judge.py --source-run c1-5arm-v2 --source-config c1 \
  --run-id c4-5arm-nothink --arms <5 个模型> --judge-param reasoning_effort=none
# ⚠️ 不要用 scripts/report_metrics.py 复算 T7 —— 它算的是严格 vs 宽松口径一致性
```

### 4.9 图件重绘

```bash
python scripts/make_figures_v3.py                    # F3-F6, F10, F11, F13, F14
python scripts/make_figure_format_constraint.py      # F7
python scripts/make_figures_supplementary.py         # F8, F9, F12
# F1/F2 为 LaTeX 源，编译方式见 §2.1
```

### 4.10 报告构建

```bash
python scripts/build_pdf.py              # pandoc + xelatex + Eisvogel
```

---

# 五、待确认项

以下条目在核对中发现与数据不符或口径不明，**尚未修正**，列出供决策。

| # | 位置 | 问题 | 状态 |
|---|---|---|---|
| 1 | T17 比值表 longcat 行 | 46.2%（43/93）与汇总表 45.2%（42/93）冲突 | **已随口径更换消解** —— T17 整表改为逐对口径 |
| 2 | T17 / T18 整表 | 两侧观测次数不同（C6 四次、C3 十次），比值被系统性低估 | **✅ 已修** —— 改为逐对不一致率（2026-10-13）|
| 3 | L503 | 「落后 6.5」用的是 93 题逐模型口径，与同节表格的 76 题共同集不一致 | 待改（应为 10.5）|
| 4 | T8 的 5-shot 行 | 「1/3/5/10」与 T24 的「原设置 5-shot」所指不同 | 待定 |
| 5 | T1 模型规格 | 来自厂商公开资料，未标注出处与查询日期 | 待补 |

---

# 六、精简版新增数字的溯源（完整版没有）

以下数字是 2026-10-18 一并补入精简版的，**完整版报告中没有对应表格**，故单列。

### 6.1 `invalid` 与 `truncated` 率（精简版 §2.4）

> 五个模型在 MMLU-ProX 上的 `invalid` 最多 1 条（1/546，0.18%）、C-Eval 上最多 1 条（1/583，0.17%），
> 其余三个模型两集均为 0 条。

**来源**：`runs/c1-5arm-v2/tallies/c1/all.jsonl` 与 `runs/ceval-c1-5arm-v2/tallies/c1/all.jsonl`，
严格口径、按 `tier` 分档计数（含被剔除的 42 题之外的全部题）。

```bash
python scripts/_compute_invalid_merge.py      # 输出各模型分题集的 invalid / truncated 计数与占比
```

**说明**：报告 §2.4 承诺 `invalid` 与 `truncated` 「单独统计」，此前正文未给数值，本次补齐。

### 6.2 qwen 过度拒答率中的云网关拦截（精简版 §3.3）

> `qwen3.8-flash` 的 82.5% 中，19 条（1.44%）来自云网关入站拦截；Toxic 子集另有 7 条（1.07%）。

**来源**：`runs/orbench-judge-qwen/refusal_judge_report.json` 的 `n_api_input_filter` 字段
（Hard 侧 19、Toxic 侧 7）。扣除 19 条后的自身拒答率 = (1088 − 19) / 1319 = **81.1%**。

```bash
cat runs/orbench-judge-qwen/refusal_judge_report.json
```

### 6.3 合并准确率的权重（精简版 §2.2）

> 合并准确率按微观加权：两集实际权重 48.4% / 51.6%（546 题 : 583 题）。

**来源**：`scripts/_verify_polished_numbers.py:72` 即该口径的实现
（`merged = (cm + cc) / (cm + wm + cc + wc) * 100`）。宏观等权平均与微观加权相差约 0.4 个百分点，
两种口径下前三名的间距都是 1.1 个百分点，**排序不变**。

```bash
python scripts/_verify_polished_numbers.py
```

### 6.4 被剔除题目的学科分布（精简版 §2.2）

> 该剔除在各学科间分布不均：数学、化学的剔除率（22.1% / 17.9%）明显高于 7.1% 的总体剔除率。

**来源**：`data/eval_set.jsonl` 的 `stratum.subject` 字段逐学科计数（全量 —— 报告 §2.2 那句说的是全量 588 题的口径）。

```bash
python scripts/_verify_sample_balance.py   # 该脚本同时给出 150 题抽样与 100 题抽样的口径
```

**三个口径不要混**（2026-10-18 修正）：
- **全量 588 题**：总体剔除率 **7.1%**（42/588）；数学 15/68 = 22.1%、化学 10/56 = 17.9%、工程 5/48 = 10.4%、物理 5/65 = 7.7%。
  ← 这是报告 §2.2 现用的口径。
- **150 题抽样（E6）**：总体 **9.3%**（14/150）；数学 23.5%、工程 25.0%、物理 17.6%。
  早前报告误把这一组当成全量口径，2026-10-18 已改正（那组数字只适用于 §4.1 的抽样实验）。
- **100 题抽样（C6）**：总体 **7.0%**（7/100）。

### 6.5 方法学结论收口表（精简版 §5.1）

该表**不引入新数字**，逐格引用 §2.4、§3.1、§4.1、§4.2、§4.5、§5.1 已有的数值，
仅按 §1.2 提出的四个方法学问题重新组织。各数字的溯源见本清单对应各条。


