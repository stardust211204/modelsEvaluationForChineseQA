# 复现指南

> 本文件收纳跑批环境、门禁读数与目录结构等操作细节。
> 想先看项目做了什么，读 [`../README.md`](../README.md)；想看结论，读
> [`../reports/modelsEvaluationForChineseQA.md`](../reports/modelsEvaluationForChineseQA.md)。

## 一、不跑批也能复算全部指标

`runs/*/tallies/`（判分台账）**已随仓库入库**，因此**不需要任何 API key** 即可复算报告里的全部统计量。
原始响应（约 75 MB）不在 git 里，另以 [Release 数据包](https://github.com/stardust211204/modelsEvaluationForChineseQA/releases/tag/v1.0.0)发布。

```powershell
# 以主评测批次为例：复算五个模型的 MMLU-ProX 准确率
.\.venv\Scripts\python.exe scripts\report_metrics.py --run-id c1-5arm-v2 --config c1
```

实测（全新 clone）：

```
deepseek-v4-pro        58.6%   deepseek-v4.1-flash  64.1%   longcat-2.5-preview-free  62.9%
mimo-v2.6-flash        49.1%   qwen3.8-flash        61.2%          （n = 546）
```

与报告 §3.1 逐个吻合。报告里每一处数字的来源与复算命令登记在
[`../reports/report_data_provenance.md`](../reports/report_data_provenance.md)。

## 二、完整流程（需要 API key）

```powershell
# 1) 环境（自建；他人 .venv 内含绝对路径，不可移植）
uv venv .venv --python 3.12
uv pip install --python ".\.venv\Scripts\python.exe" `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

# 2) 密钥
Copy-Item .env.example .env    # 填入真实 key

# 3) 跑批前探针（前置步骤，结果回填 docs/param_matrix.md）
.\.venv\Scripts\python.exe scripts\smoke_arms.py
.\.venv\Scripts\python.exe scripts\probe_params.py

# 4) 单测 + 两个门禁
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\ci_gate.py      # 210 项
.\.venv\Scripts\python.exe scripts\exp_smoke.py    # 124 项

# 5) 出报告（Markdown → PDF；默认源即对外报告）
.\.venv\Scripts\python.exe scripts\build_pdf.py

# 6) 报告自检（不需要 API）
.\.venv\Scripts\python.exe scripts\_check_polished_report.py
.\.venv\Scripts\python.exe scripts\_check_render_safe.py
```

**小样本端到端演练**：`.\run_all.ps1`（末步即门禁），用极小样本走完「造题 → 推理 → 判分 → 门禁」。

**报告排版**：`build_pdf.py` 一键编译 PDF（pandoc + xelatex + Eisvogel），参数取自 CCL 2026
官方中文论文模板。

## 三、门禁在两种工作区状态下的读数差异

差别只在 `raw` 在不在（原始响应包走 Release，不入库）：

| 场景 | `ci_gate`（210 项） | `exp_smoke`（124 项） |
| --- | --- | --- |
| **完整工作区**（raw 已下载） | PASS **203** / FAIL **1** / WARN 6 | **124/124** |
| **刚 clone**（raw 未下载） | PASS 162 / FAIL **0** / **SKIP 42** | **123/123**（跳过 1 项） |

被跳过的是**输入不存在**的检查（「raw 完整性」「judge tally 完整性」「被复核 run 含 5 个模型」等），
脚本会写明原因 —— **这些是跳过项，不计为失败**。

完整工作区的那 1 项 FAIL 是 `longcat` 的**上游连接失败**（间歇性路由缺陷，客户端侧无法修复），
与判分口径无关；来历见 [`../README.md`](../README.md) §四「③ 单元测试与回归门禁」。

读数判据是**全局**的：只要有一个 `runs/*/raw` 存在，就按完整工作区标准判 ——
此时某个批次缺 raw 视为破坏性变更，直接判 FAIL。要拿到完整读数，
把 [Release 里的 raw 包](https://github.com/stardust211204/modelsEvaluationForChineseQA/releases/tag/v1.0.0)解压到仓库根（保持 `runs/{run_id}/raw/...` 结构）即可。

## 四、目录结构

```
src/meval/       管线本体（见 README §二）
scripts/         入口、实验与分析脚本、核校工具（66 个 .py）
configs/         arms.yaml（五个模型 + 各自生效参数）· matrix.yaml · exclusions.json
data/            eval_set.jsonl · ceval_set.jsonl · orbench_set.jsonl · frozen.json
prompts/         c1_strict / c2_free / c5_fewshot / e2_labelshuffle / e6_* / judge_*
docs/            design.md · limitations.md · param_matrix.md · badcase_method.md
figures/         14 张正式图 + latex/（管线图源码与 HOWTO.md）
runs/            每次跑批：manifest + raw（未入库） + tallies（已入库，可直接复算）
reports/         对外报告 · 溯源清单 · 批次台账 · 专项实验报告
tests/           13 个模块 / 182 项（168 个测试函数，14 项来自参数化展开）
```
