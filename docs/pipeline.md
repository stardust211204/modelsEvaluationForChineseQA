# 管线运营

> 面向执行。设计理由见 `design.md`，局限见 `limitations.md`。

---

## 0. 一次完整跑批的全部命令

```powershell
cd <workspace>\modelsEvaluation
$py = ".\.venv\Scripts\python.exe"

# ① 冒烟：确认三个模型连通
& $py scripts\smoke_arms.py

# ② 参数探测：逐模型确认温度/思考开关/max_tokens（★ 必做，结果回填 configs\arms.yaml）
& $py scripts\probe_params.py

# ③ 伪别名指纹：确认三个模型确为不同模型
& $py scripts\probe_distinct.py

# ④ 单测
& $py -m pytest

# ⑤ 造题：难度代理打分 → 分层抽样 → 冻结
& $py -m meval.build_set        # TODO 待实现 CLI
& $py -m meval.adapt            # TODO
& $py -m meval.execute --config c1
& $py -m meval.tally --run-id <rid>
& $py -m meval.metrics --run-id <rid>

# ⑥ 完整复现
.\run_all.ps1                   # TODO
```

---

## 1. 执行顺序与依赖

```
arms.yaml 参数回填 ──┐
                     ├─> ③ 造题（eval_set.jsonl 冻结 + 哈希）──┐
探针通过 ────────────┘                                        │
                                                              ▼
                                        C1 全量 ─> C2 ─> C3 ─> C5
                                                              │
                                                              ▼
                                          C4 judge 复核 ─> 指标层
                                                              │
                                                              ▼
                                                    报告 + 三图
```

**关键路径**：`③ 造题 → C1 全量 → 指标层 → 报告`。C1 是唯一的长串行段（1,800 次调用），它的耗时决定后面所有环节的余量。

---

## 2. 每一步的验收标准

| 步骤 | 验收 |
|---|---|
| ① 冒烟 | □ 三个模型各 3 题 HTTP 200　□ 无空 `content`（若空 → 回 ② 调 `max_tokens`/思考开关） |
| ② 参数探测 | □ 每个模型记录：温度是否生效、思考关闭用哪个参数、`max_tokens` 下限　□ 结果回填 `arms.yaml` + `docs/param_matrix.md` |
| ③ 别名指纹 | □ 三个模型两两答案完全一致率 **< 80%**（且已剔除空响应） |
| ④ 单测 | □ 全绿。**判分单测是唯一能拦住「静默空 content」的地方，不能跳过** |
| ⑤ 造题 | □ `eval_set.jsonl` 588 条　□ 哈希已记入 manifest　□ **抽样保真度自检**通过（学科/选项数 max_gap ≤15pp） |
| C1 跑批 | □ 1,800 条全部落盘　□ `truncated` 率 < 5%（否则说明 `max_tokens` 仍偏小） |
| 指标层 | □ 每个准确率都带 95% CI　□ `truncated` / `invalid` / 准确率三者分账 |
| 报告 | □ Limitations 齐全　□ 三图落盘　□ 干净目录按 README 可复现 |

---

## 3. 跑批中会遇到的四类情况（及处理）

| 情况 | 判据 | 处理 |
|---|---|---|
| **退避重试** | 429 → `ErrorKind.QUOTA`（重试 4 次，2s 起指数退避）<br>超时 / 5xx → `ErrorKind.TRANSPORT`（1s 起） | 自动，记录在 raw 的 `error` 字段 |
| **不重试** | 4xx（非 429）→ `ErrorKind.CLIENT` | 直接落盘记账，**不要盲目重试**（重试不会变好，只烧额度） |
| **断点续跑** | 中途 Ctrl-C 后重跑同一命令 | 扫描已有 raw → 构建已完成键集合 → 跳过。**已完成条目零重复调用** |
| **静默失败** | HTTP 200 但 `content` 为空、`finish_reason='length'` | 判为 `truncated`（**不是** `invalid`）。若大面积出现 → 提高 `max_tokens` 或换思考关闭参数 |

---

## 4. 口径冻结清单（跑批前逐项打勾，跑批中不得改动）

- [ ] `configs/arms.yaml` 的每个模型 `params` 已由探测回填
- [ ] prompt 模板已定稿（`prompts/c1_strict.txt` 等）
- [ ] 解析阶梯与四档判级已单测通过
- [ ] `eval_set.jsonl` 已冻结并记录哈希
- [ ] C5 的 5 条 few-shot 示例已从 **validation** 选定并固定顺序（写进 manifest）
- [ ] C4 的 judge 模型已选定，且**与被评模型不同家族**
- [ ] C2 / C5 的题集已确认是 C1 的子集

> **跑批中途改任何一个，前面的数据作废。** 要改就新开一个 `run_id`。

---

## 5. 产物清单

| 产物 | 路径 | 是否入库 |
|---|---|---|
| 跑批元数据 | `runs/{run_id}/manifest.json` | 建议入库（体积小，含参数生效表与哈希） |
| 原始调用事实 | `runs/{run_id}/raw/{config}/{arm}.jsonl` | 不入库（体积大）；抽样保留到 `data/raw_evidence/` |
| 判分结论 | `runs/{run_id}/tallies/{config}/{arm}.jsonl` | 不入库（可由 raw 重算） |
| 指标汇总 | `runs/{run_id}/metrics.csv` | **入库**（报告的数据来源） |
| 复核记录 | `runs/{run_id}/audit.json` | 入库（一致率 + Kappa） |
| 错题归因 | `runs/{run_id}/badcase.csv` | 入库（80–120 条人工标注） |
| 三张图 | `figures/*.png` | 入库 |
| 评测报告 | `reports/eval_report.md` | 入库 |

---

## 6. 常见返工及其成本（这就是 raw/tally 分离的价值）

| 返工类型 | 需要重跑什么 | 成本 |
|---|---|---|
| 改解析正则 / 档位边界 | **只重跑 `tally` + `metrics`** | 秒级～分钟级 |
| 改归因判据 | 只重跑 `metrics` | 秒级 |
| 改 judge prompt | 只重跑 C4（150 次调用） | 分钟级 |
| 改主 prompt 模板 | **必须重跑该配置的全部调用** | C1 = 1,800 次 |
| 改 `max_tokens` | 必须重跑（影响输出） | 全量 |

> 所以：**prompt 与参数要在 F2 之前冻结**，判分逻辑可以放心地晚一点再打磨。
