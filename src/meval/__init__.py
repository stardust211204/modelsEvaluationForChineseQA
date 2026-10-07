"""meval —— 大模型准确性评测管线。

分层（各层职责边界见 docs/design.md §3）：
    ① build_set  造题          Instance[]
    ② adapt      请求构造      Instance × Config × Arm → PromptSpec
    ③ execute    执行          PromptSpec → ModelResult   （唯一发 HTTP 的地方）
    ④ tally      判分          ModelResult → Tally        （不发任何请求）
    ⑤ metrics    聚合          Tally → Stat
    ⑥ report     报告          Stat → 图 + 文档

核心工程决策：raw 与 tallies 分离 —— 改判分规则不需要重跑 API。
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import paths  # noqa: F401
from .types import (  # noqa: F401
    Attribution,
    Config,
    ErrorKind,
    Instance,
    ModelResult,
    ParseMode,
    PromptSpec,
    RunManifest,
    Source,
    Split,
    Stat,
    Tally,
    Tier,
)

__all__ = [
    "paths",
    "Instance",
    "PromptSpec",
    "ModelResult",
    "Tally",
    "Stat",
    "RunManifest",
    "Config",
    "Split",
    "Source",
    "Tier",
    "ParseMode",
    "Attribution",
    "ErrorKind",
]
