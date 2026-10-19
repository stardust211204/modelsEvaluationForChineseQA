# 第三方件与致谢

> 本项目**自己写的代码与文档**以 **MPL-2.0**（Mozilla Public License 2.0）授权（见仓库根目录 `LICENSE`）。
> 但仓库里还包含若干**第三方件**与**第三方数据**，它们**不适用**该授权，
> 各自保留上游协议 —— 本文件逐项列出，以便合规复用。

---

## 1 随仓库分发的第三方代码 / 模板

| 名称 | 版本 | 许可 | 许可证文件 | 用途 |
| --- | --- | --- | --- | --- |
| **Eisvogel** | 3.5.1 | **BSD-3-Clause** | [`reports/Eisvogel-3.5.1/LICENSE`](reports/Eisvogel-3.5.1/LICENSE) | 报告 PDF 的 LaTeX 模板 |
| **PlotNeuralNet** | 1.0.0 | **MIT** | [`reports/PlotNeuralNet-1.0.0/LICENSE`](reports/PlotNeuralNet-1.0.0/LICENSE) | 早期版管线图的绘制工具（当前版本已改为独立 TikZ） |

**兼容性说明**：BSD-3-Clause 与 MIT 都是宽松协议，可以并入以 MPL-2.0 授权的作品
（MPL 的义务只及于被修改的 MPL 文件，不延伸到这些第三方件）；二者的许可证原文随本仓库一并保留，
满足「保留版权声明与免责声明」的要求。

> **Eisvogel 特别说明**：本仓库只随附**模板本体**（`eisvogel.latex`，29 KB）与其
> LICENSE。上游 30 MB 的整包（`examples/`、`document.pdf`、`preview.png` 等）
> 已被 `.gitignore` 排除，本项目不需要它。
> （早前版本的 `.gitignore` 注释里把它的协议误标为 LPPL-1.3c，已更正为 BSD-3-Clause。）

---

## 2 第三方数据集

数据集**不在** `THIRD_PARTY` 的代码范畴内，且各自的协议差异较大
（其中 C-Eval 含 **NonCommercial + ShareAlike** 硬约束）。逐份清单与说明见：

➡️ **[`data/README.md`](data/README.md)**

| 数据集 | 上游许可 |
| --- | --- |
| MMLU-ProX（经 `MMLU-ProX-Lite` 切片） | MIT |
| C-Eval | **CC-BY-NC-SA-4.0** |
| OR-Bench | CC-BY-4.0 |

---

## 3 运行期依赖（不随仓库分发）

`pyproject.toml` 声明的第三方 Python 依赖由 pip 在安装时获取，各自保留其上游协议。
主要几项：

| 依赖 | 用途 |
| --- | --- |
| `openai` | 各厂商的 OpenAI 兼容 API 客户端 |
| `datasets` / `huggingface_hub` | 题集获取 |
| `pyyaml` / `jsonlines` | 配置与台账读写 |
| `pytest` | 单测与门禁 |
| `matplotlib` | 图件绘制 |

报告 PDF 的编译链另需外部工具（**pandoc** + **XeLaTeX**/TeX Live），
它们不随仓库分发。

---

## 4 引用与致谢

本报告的方法学与实验设计受以下工作启发，正文 §6.3 有完整出处：

- **HELM** —— 场景 / 适配 / 指标三元组（arXiv:2211.09110，TMLR 2023）
- **MMLU-ProX**（arXiv:2503.10497，EMNLP 2025）· **C-Eval**（arXiv:2305.08322，NeurIPS 2023）·
  **OR-Bench**（arXiv:2405.20947，ICML 2025）
- **Tam et al. 2024**，*Let Me Speak Freely?*（arXiv:2408.02442，EMNLP 2024 Industry Track）
- **Lee et al. 2026**，*The Format Tax*（arXiv:2604.03616）
- **Sprague et al. 2025**，*To CoT or not to CoT?*（arXiv:2409.12183，ICLR 2025）

本项目未修改上述数据集的内容，仅在格式整理、抽样与统计口径上做了处理；
所报告的一切数字均可在 `runs/` 的判分台账上离线复算。

---

*若发现本文件的协议标注与上游最新声明不符，以上游为准，并欢迎提 issue 更正。*
