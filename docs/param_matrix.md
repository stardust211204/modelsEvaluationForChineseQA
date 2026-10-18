# 实测参数矩阵

> 来源：`ModelsEvaluationBasedOnHELM` 的 F0 探测，**116 次实跑**，原始记录见 `data/raw_evidence/`
> 结论口径：**「参数被接受（HTTP 200）」≠「参数生效」**——本表只记录**实测生效**的行为

---

## 1. 池子清点（`GET /models`）

快照日期：2026-09-28。**选模型前请重新拉取确认。**

| 家族 | 数量 | 模型 ID |
|---|---|---|
| DeepSeek | 5 | `deepseek-flash` · `deepseek-v4-flash` · `deepseek-v4-flash-vision-exp` · `deepseek-v4.1-flash` · `deepseek-v4-pro` |
| 智谱 GLM | 4 | `glm-5.1` · `glm-5.2` · `glm-5.3` · `glm-5.3-flash` |
| 通义千问 | 5 | `qwen3.6-plus` · `qwen3.7-plus` · `qwen3.7-max` · `qwen3.8-flash` · `qwen3.8-max` |
| 月之暗面 Kimi | 3 | `kimi-k2.6` · `kimi-k2.7-code` · `kimi-k3` |
| MiMo | 4 | `mimo-v2.5` · `mimo-v2.5-pro` · `mimo-v2.6-flash` · `mimo-v2.6-pro` |
| MiniMax | 3 | `minimax-m2.5` · `minimax-m2.7` · `minimax-m3` |
| xAI Grok | 2 | `grok-4.6` · `grok-4.7` |
| LongCat | 2 | `longcat-2.0` · `longcat-2.5-preview-free` |
| 代号未知 | 8 | `gpt-5.6-luna` · `gpt-6-luna` · `hy3` · `hy4-preview` · `omen-alpha` · `space-bunny-free` · `muse-spark-1.2-contributor` · `muse-spark-1.3-contributor` |

**合计 36 个。**

---

## 2. ⚠️ DeepSeek 家族存在别名（选模型必读）

| ID | 官方口径 | 可作独立模型 |
|---|---|---|
| `deepseek-flash` | 正式名 → DeepSeek-V4.1-Flash | ✅ |
| `deepseek-v4.1-flash` | 同一模型的别名 | ❌ |
| `deepseek-v4-flash` | **旧名，模型已下线**，请求被路由到 V4.1-Flash | ❌ |
| `deepseek-v4-flash-vision-exp` | 官方标注已下线，同样路由 | ❌ |
| `deepseek-v4-pro` | 不同模型（V4-Pro-0813） | ✅ |

**DeepSeek 家族实际只有 2 个可用模型。** 原方案把 `deepseek-flash` 与 `deepseek-v4-flash` 当"同厂两代对照"，两个模型会跑出几乎相同的分数——代际差是伪结论。

---

## 3. 参数接受度

8 种参数组合 × 4 个模型 = 32 次调用，**全部返回 HTTP 200，无参数被拒绝**：

| 参数 | 结果 |
|---|---|
| `temperature` 0 / 0.1 / 1.0 | 全部 200 |
| `top_p=0.9` | 全部 200 |
| `seed=42` | 全部 200 |
| `thinking={"type":"disabled"}` | 全部 200 |
| `reasoning_effort="none"` | 全部 200 |

> ⚠️ **这不能说明任何事**。聚合端点对未知参数的习惯做法是**静默忽略**。

---

## 4. 温度是否真正生效

同题 × 6 次 @ `temperature=1.0`，统计不同输出数：

| 模型 | 不同输出 | 判定 |
|---|---|---|
| `deepseek-v4-flash` | **1 / 6** | ⚠️ **温度被忽略（完全确定）** |
| `deepseek-v4-flash-vision-exp` | 2 / 6 | 弱生效 |
| `deepseek-flash` | 2 / 6 | 弱生效 |
| `deepseek-v4.1-flash` | 2 / 6 | 弱生效 |

**结论：温度在本池子上弱生效甚至无效。** 因此"固定温度 = 可复现"**不成立**，C3 重复采样模型是唯一能量化确定性的手段。

---

## 5. 思考关闭：按模型不同，且会冲突 ★

用真实评测 prompt（含"只回单个大写字母"强约束），观察 `reasoning_content` 长度：

| 模型 | 无参数（基线） | `reasoning_effort:"none"` | `thinking:{"type":"disabled"}` | 两者同给 |
|---|---|---|---|---|
| `deepseek-flash` | reasoning 1,131 字 | ✅ **0 字** | ✅ 0 字 | ✅ 0 字 |
| `deepseek-v4-flash` | reasoning 1,097 字 | ✅ **0 字** | ❌ **仍 4,976 字**（反而更长） | ✅ 0 字 |
| `glm-5.3-flash` | reasoning 106 字 | ❌ **思考跑进 `content`**（408 tokens） | ❌ 仍 125 字 | 🔴 **HTTP 400** |

**三条结论**：
1. `reasoning_effort:"none"` 对 DeepSeek 两个模型有效；**对 GLM 不仅无效，还把思考倒进 `content`**——直接破坏格式遵循
2. `thinking:{"type":"disabled"}` 对 `deepseek-v4-flash` **无效**（思考从 1,097 涨到 4,976 字）
3. GLM 同时收到两者 **HTTP 400** → 参数**互斥**

### 顺带的一个实锤

`deepseek-v4-flash` 同一题：`reasoning_effort=none` 时答 **H**，`thinking=disabled`（无效、仍在思考）时答 **I**。
→ **同模型、同题，仅思考配置不同就换了答案。** 这是"思考开关会改变结论"的直接证据。

---

## 6. 空 `content` 的真实成因（重要）★

| 模型 | `max_tokens` | `finish_reason` | `content` 长度 | `reasoning_content` 长度 |
|---|---|---|---|---|
| `deepseek-v4-flash` | 64 | stop | 1（`'I'`） | 4,759 |
| `deepseek-flash` | 64 | **length** | **0** | 278 |
| `deepseek-flash` | 512 | **length** | **0** | **1,955** |
| `glm-5.3-flash` | 64 | **length** | **0** | 264 |
| `glm-5.3-flash` | 512 | stop | 1（`'I'`） | 1,159 |

**机制**：这些模型把思考写进**独立的 `reasoning_content` 字段**（`message` 的 keys 为 `content` + `reasoning_content` + `role`）。当思考耗尽 `max_tokens` 时，`finish_reason='length'` 且 **`content` 为空字符串**，而 **HTTP 状态是 200**。

**后果**：若只读 `message.content`，会得到空串 → 判分阶梯全失败 → 记为"无效输出"，而**实际上模型答对了**（答案在 reasoning 里）。这会把"思考长度差异"误报成"能力差异"。

**应对（已写入设计）**：
1. `max_tokens` **从 512 提到 2048**（GLM 思考实测 519 token 就已超 512）
2. **同时记录 `content` 与 `reasoning_content`**，并保存 `finish_reason`、`reasoning_tokens`
3. 判分档位新增 **`truncated`**，与 `invalid` 分离

---

## 7. 伪别名检测：`resp_model` 不作证 ★

| 检查 | 结果 |
|---|---|
| 请求 `model` vs 响应 `model` 字段 | **完全一致（回显）**，四个模型逐一验证 |
| 结论 | **该字段不含任何后端信息，不能用于别名检测** |

**替代方案：行为指纹**
- 挑 30–40 道有区分度的题，各模型全跑
- 两个模型答案完全一致率 **≥ 80%** → 高度可疑（真别名应接近 100%）
- ⚠️ **必须先剔除空响应**：本次探测中出现过三个模型 93%–100% 一致的假象，经查**全部由共同返回空 `content` 造成**（即 §6 的现象）

---

## 8. 三模型阶段的实测结论（2026-10-03 回填）—— 历史快照

> **⏱ 时效（2026-10-17 补）**：本节是**三模型阶段**的探测快照，表里只有 3 个模型。
> 项目最终为 **5 个正式模型**，各模型的**冻结参数以 [`../configs/arms.yaml`](../configs/arms.yaml) 为准**
> —— 新增了 `qwen3.8-flash` 与 `deepseek-v4-pro`（后者走官方端点，与其余四个模型不同），
> 另有 `space-bunny-free` 因**无法关闭思考**、与其余模型口径不同而被列入 `excluded`。
>
> **本条以下的方法学结论仍然有效，且都已被后续实现吸收：**
>
> 1. **温度被忽略 ⇒ 重复采样是唯一能量化确定性的手段**。
>    后续实测把这条推得更远：参数被忽略**不等于**输出可复现 ——
>    同题重复 10 次，答案全同率最低的模型只有 **28.0%**（报告 §3.2）。
> 2. **关思考参数跨厂商不可移植** —— LongCat 只认 `thinking`、DeepSeek 与小米只认
>    `reasoning_effort`，且两者同给会返回 **400**。
>    这就是「放弃全局统一参数、改为按模型探测后冻结」的实证依据（`design.md` §6）。
> 3. **厂商私有参数必须走 `extra_body`** —— 直传 kwarg 会在**本地**抛 `TypeError`，
>    错误信息却像网络问题。已固化为 `tests/test_client_params.py`，
>    并记入 README §三 缺陷 6。

已选定并**完成逐模型探测**（39 次调用，原始记录 `data/raw_evidence/probe_arms.jsonl`）：

| 模型 | 厂商 | 温度生效 | 关思考参数（实测有效） | `max_tokens` 下限 | 冻结 params |
|---|---|---|---|---|---|
| `longcat-2.5-preview-free` | 美团 | ❌ 被忽略 | **`thinking: {type: disabled}`** | 512 | `max_tokens=1024, temperature=0.1, thinking={type:disabled}` |
| `mimo-v2.6-flash` | 小米 | ❌ 被忽略 | **`reasoning_effort: "none"`** | 512 | `max_tokens=1024, temperature=0.1, reasoning_effort=none` |
| `deepseek-v4.1-flash` | DeepSeek | ❌ 被忽略 | **`reasoning_effort: "none"`** | 512 | `max_tokens=1024, temperature=0.1, reasoning_effort=none` |

**三条结论**：

1. **三个模型温度全部失效**——同题 ×6 @ `temperature=1.0` 均只产生 **1 种**输出。
   → **C3 重复采样模型的价值因此确定**：它是唯一能量化"我们的口径到底有多确定"的手段。
2. **关思考参数跨厂商不可移植**——LongCat 只认 `thinking`，另外两个模型只认 `reasoning_effort`。
   → 这正是"参数按模型配置"的实证依据。
3. `max_tokens` 512 即够；实际取 **1024**，为隐含思考留余量。

### ★ 探测中发现的一个致命工程坑（已修复）

首次探测时 LongCat 的两个变体返回 `http_status=None`、`error_kind=transport`，看起来像网络问题。
查原始记录发现真实原因是**本地**异常：

```
TypeError: Completions.create() got an unexpected keyword argument 'thinking'
```

**OpenAI SDK 的 `chat.completions.create()` 只接受它自己声明的参数名。**
厂商私有参数（`thinking` / `reasoning_effort` 等）**直接当 kwarg 传会在本地就失败，根本到不了服务端**，
而错误信息极具误导性。

**修复**：`client.py` 增加参数分流 —— 标准参数走 kwargs，私有参数走 **`extra_body`** 透传。
已加单测固化（`tests/test_client_params.py`），并把 `TypeError/ValueError/AttributeError` 归为
`ErrorKind.CLIENT`（**不可重试**，否则会白烧重试与额度）。

### 三个模型互异性指纹（通过）

`scripts/probe_distinct.py`：30 题 × 3 个模型 = 90 次调用，三个模型均 30/30 有效作答。

| 对比 | 完全一致率 | 判定 |
|---|---|---|
| LongCat vs MiMo | 13/30 = **43.3%** | 通过 |
| LongCat vs DeepSeek | 16/30 = **53.3%** | 通过 |
| MiMo vs DeepSeek | 12/30 = **40.0%** | 通过 |

**阈值 80%，无一对接近** → 三个模型确为三个不同模型，可用于对比。

### 仍需留意

| # | 风险 |
|---|---|
| 1 | `longcat-2.5-preview-free` 名字里带 `preview` 与 `free`：**可能限流、可能随时变更或下线**。跑批中途换后端 = 数据不可比；建议跑批期间记录 `request_datetime` 并事后比对前后半段准确率 |
| 2 | `mimo-v2.6-flash` 为免费档，限流未知 |
