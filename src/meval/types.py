"""核心数据模型。

六类，各司其职，互不越界：
    Instance     一道题（不含任何模型信息）
    PromptSpec   一次请求的输入
    ModelResult  一次调用的返回
    Tally        一次判分结论（可对同一 ModelResult 反复重算）
    Stat         聚合量
    RunManifest  一次跑批的元数据

设计要点：
  * ModelResult 与 Tally 分离 —— 改判分规则不需要重跑 API
  * ModelResult 保留 reasoning / finish_reason / reasoning_tokens
    —— 推理模型把思考写进独立字段，且思考与答案共享 token 预算
  * Instance.gold_indices 是列表 —— 支持一题多正解（借 HELM 的 Reference.tags 机制）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------

class Split(str, Enum):
    """数据划分。few-shot 示例必须取自 VALID，不得取自 TEST（否则数据泄漏）。"""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Source(str, Enum):
    """题目来源。"""
    PUBLIC = "public"


class Config(str, Enum):
    """配置矩阵。每个配置只改一个因子，C1 是干净基线。"""
    C1 = "c1"  # 基线：强格式约束 + 0-shot + 关思考
    C2 = "c2"  # 去掉格式约束（测判分口径 / 风格偏差）
    C3 = "c3"  # 重复采样（测确定性）
    C4 = "c4"  # LLM-as-Judge 复核
    C5 = "c5"  # 5-shot（测示例数量）
    # ★ C5b：5-shot 的第二组示例 —— 只改「示例的选项数」，示例数量仍是 5。
    #   动机：C5 在两个评测集上共用一组示例（选项数均值 8.4），
    #   而 C-Eval 全部 583 题都是 4 选项 ⇒ 在 C-Eval 上成了跨格式迁移实验。
    #   本配置让示例选项数与目标题匹配（MMLU 10 项 / C-Eval 均值 5.0），
    #   用以分离「示例数量」与「示例格式错配」两个因子。
    C5B = "c5b"
    C6 = "c6"  # 选项乱序（测位置敏感性）—— rep 即「第几个排列」
    C7 = "c7"  # 拒答/弃权评测（双向：过度拒答 vs 该拒未拒）


class Tier(str, Enum):
    """判分档位。前两档计入准确率分母，后两档独立报告。

    invalid 与 truncated 必须分开：
        invalid   大致归因于「模型」（不听话、格式乱）
        truncated 归因于「我们的配置」（token 预算不够 / 未能关掉思考）

    拒答档位（refuse_* / comply_unsafe / partial）只用于**拒答评测**，
    不参与 C1–C6 的准确率。它们回答的是另一个问题：
    「该不该答」而不是「答得对不对」。
    """
    CORRECT = "correct"
    WRONG = "wrong"
    INVALID = "invalid"
    TRUNCATED = "truncated"

    # ---- 拒答评测专用（C7）------------------------------------------------
    # ★ 只增不改：C1–C6 的判分路径一行都没动。
    REFUSE_OK = "refuse_ok"          # 该拒且拒了        —— 正向
    REFUSE_OVER = "refuse_over"      # 不该拒却拒了      —— 过度拒答
    COMPLY_UNSAFE = "comply_unsafe"  # 该拒却答了        —— 最严重
    PARTIAL = "partial"              # 半拒半答 / 加了免责但给了实质内容


class ParseMode(str, Enum):
    """双解析口径。同一份 raw 用两套口径各判一次。"""
    STRICT = "strict"    # 全文匹配单个大写字母
    LENIENT = "lenient"  # 三级阶梯：bare -> answer 句式 -> 末行孤立字母


class Attribution(str, Enum):
    """归因类别。只作用于 Tier.WRONG。"""
    FACTUAL = "factual"
    REASONING = "reasoning"
    OPTION_CONFUSION = "option_confusion"


class ErrorKind(str, Enum):
    """错误分级（借 HELM 的 error_flags 思路）。决定是否重试。"""
    TRANSPORT = "transport"  # 网络 / 超时 —— 可重试
    QUOTA = "quota"          # 429 —— 退避重试
    CLIENT = "client"        # 其它 4xx —— 不重试，直接记账


# --------------------------------------------------------------------------
# ① Instance —— 一道题
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """一道题。不含任何模型相关信息。"""

    id: str
    question: str
    options: list[str]
    gold_indices: list[int]          # 列表：支持一题多正解
    split: Split = Split.TEST
    stratum: dict[str, str] = field(default_factory=dict)  # 学科/难度档/选项数档
    source: Source = Source.PUBLIC
    note: str = ""                   # 自建题的出题意图

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def gold_letters(self) -> list[str]:
        return [chr(65 + i) for i in self.gold_indices]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["split"] = self.split.value
        d["source"] = self.source.value
        return d


# --------------------------------------------------------------------------
# ② PromptSpec —— 一次请求的输入
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSpec:
    """一次请求的输入。params 是该臂「实测生效后」冻结的字典，不是全局常量。"""

    run_id: str
    config: Config
    arm: str
    instance_id: str
    rep: int = 0                     # 重复序号（C3 用，其余为 0）
    prompt_text: str = ""
    demos: list[str] = field(default_factory=list)  # 本次用到的 few-shot 题 id
    params: dict[str, Any] = field(default_factory=dict)
    # judge 专用：这条请求在**评审哪个臂的输出**（非 judge 场景为空串）。
    # ★ 必须进去重键：C4 要对三臂的同一批题各判一次，若键里只有 judge 自己的
    #   名字，三臂的请求会互相覆盖、只剩最后一个臂
    #   —— 与 docs/HANDOFF.md §4 坑 6（"结果字典键必须含 arm"）是同一类错误。
    target_arm: str = ""

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        """断点续跑的去重键。"""
        return (self.config.value, self.arm, self.instance_id, self.rep, self.target_arm)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.value
        return d


# --------------------------------------------------------------------------
# ③ ModelResult —— 一次调用的返回
# --------------------------------------------------------------------------

@dataclass
class ModelResult:
    """一次调用的返回。

    为推理模型扩展了三个字段：
        reasoning         思考内容（独立字段，不占 content）
        finish_reason     自然结束 / 命中长度上限 / 命中 stop
        reasoning_tokens  思考消耗的 token 数
    request_datetime 保留 HELM 的原始理由：API 背后的模型可能变化。
    """

    instance_id: str
    arm: str
    config: Config
    rep: int = 0
    target_arm: str = ""             # judge 专用：本条评审的是哪个臂的输出

    success: bool = False
    http_status: int | None = None
    error: str = ""
    error_kind: ErrorKind | None = None

    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""          # stop / length / ...
    reasoning_tokens: int = 0

    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    request_datetime: str = ""       # ISO8601

    resp_model: str = ""             # 响应回显的 model —— 注意：只是回显，不含后端信息
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.value
        d["error_kind"] = self.error_kind.value if self.error_kind else None
        return d


# --------------------------------------------------------------------------
# ④ Tally —— 一次判分结论
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tally:
    """一次判分结论。与 ModelResult 分离，可对同一份 raw 用不同口径重算。"""

    instance_id: str
    arm: str
    config: Config
    rep: int = 0
    mode: ParseMode = ParseMode.STRICT

    tier: Tier = Tier.INVALID
    answer_index: int | None = None
    tier_step: str = ""              # 命中解析阶梯的哪一级
    invalid_reason: str = ""
    attribution: Attribution | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.value
        d["mode"] = self.mode.value
        d["tier"] = self.tier.value
        d["attribution"] = self.attribution.value if self.attribution else None
        return d


# --------------------------------------------------------------------------
# ⑤ Stat —— 聚合量
# --------------------------------------------------------------------------

@dataclass
class Stat:
    """聚合量。沿用 HELM Stat 的可累加思路，加 bootstrap CI 与分层拆分。"""

    name: str
    config: Config
    arm: str

    n: int = 0
    n_correct: int = 0
    n_invalid: int = 0
    n_truncated: int = 0

    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    by_stratum: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.value
        return d


# --------------------------------------------------------------------------
# ⑥ RunManifest —— 跑批元数据
# --------------------------------------------------------------------------

@dataclass
class RunManifest:
    """一次跑批的元数据。让每个数字可回溯到哪次跑批、哪版代码、哪版数据。"""

    run_id: str
    configs: list[str] = field(default_factory=list)
    arms: list[str] = field(default_factory=list)
    params_effective: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_hash: str = ""
    # 每个配置的 prompt 指纹（题集切片 + 模板 + few-shot 示例）。
    # 断点续跑的键不含 prompt，改模板后沿用同一 run_id 会静默复用旧数据 ——
    # 记在 manifest 里，resume 时对不上就拒跑。
    prompt_fingerprints: dict[str, str] = field(default_factory=dict)
    # 本 run 显式覆盖的臂参数（如 max_tokens=2048 的稳健性重跑）。
    # 门禁拿它来解释"为什么这个 run 的参数与 arms.yaml 不同" ——
    # 有声明就是有意的变体，没声明才是参数漂移。
    param_overrides: dict[str, Any] = field(default_factory=dict)
    code_version: str = ""
    models_snapshot: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    note: str = ""
    # 本次跑批用的**哪份题集**。默认 eval_set（data/eval_set.jsonl）。
    # ★ C7 拒答评测用的是 data/refusal_set.jsonl —— 不声明的话，
    #   门禁会拿它的哈希去比冻结的 eval_set 哈希，把"用了另一份数据集"
    #   误报成"题集被改了，此 run 不可配对比较"。
    dataset: str = "eval_set"
    # ---- 以下两项为**补充验证实验**新增，默认 None ⇒ 历史 manifest 不受影响 ----
    #   `reps`：本 run **实际**用的重复次数。门禁原先只读 matrix.yaml 的 reps，
    #     于是 `--reps 1` 的 run 会被按 matrix 的 reps（如 c3 的 10）判成"缺条数"。
    #   `param_arms`：`--param` **实际作用的臂**。原先 param_overrides 被当作
    #     作用于全部臂；用 `--param-arms` 限定后，未覆盖的臂会被误报成参数漂移。
    #   两项都是"把有意的变体如实声明出来"，与 param_overrides 的用途一致。
    reps: int | None = None
    param_arms: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
