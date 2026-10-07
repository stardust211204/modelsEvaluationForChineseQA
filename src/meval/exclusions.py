"""题集排除清单：把**已确认有质量缺陷**的题从统计口径里剔掉，但**不动题集本身**。

为什么是"分析层过滤"而不是"改题集"
    `data/eval_set.jsonl` 的 sha256 被冻结在 `data/frozen.json`，每个 run 的
    manifest 都记着自己的 `data_hash`，门禁逐条比对。**删题会改哈希** ⇒
    所有历史 run 立刻被判为「题集已变，不可与其它 run 配对比较」——
    8,000+ 次调用的可比性一夜归零。

    所以排除清单是**独立制品**：
      * `data/eval_set.jsonl`  一字不动，哈希不变，历史 run 全部有效
      * `configs/exclusions.json` 记录被剔除的题 + 依据 + 生成时间
      * 分析层读它并过滤，`raw/` 与 `tallies/` 完整保留 ⇒ **随时可回溯、可撤销**

剔除依据（2026-10-05）
    MMLU-ProX 的中文版是从英文 PDF 抽取+翻译而来，**公式密集题存在 LaTeX 提取错误**，
    实测四类模式（见 `detect_bad_symbols`）：

        \\pir     应为 \\pi r        —— 命令后丢空格，36 题
        ^\\pi     上标缺失或嵌套错误             7 题
        x\\alpha  字母后紧跟命令无空格           6 题
        $ $       空数学环境                     7 题

    这些题**在源头就是坏的**，模型答不对不能归因于能力。
    ★ 与 `longcat` 在公式题上 22.2% 判不出那件事**不是同一回事**：
      那条是模型行为特征（保留，进画像卡）；这条是题目缺陷（剔除）。

用法
    from meval import exclusions
    drop = exclusions.load()                    # set[str]
    keep = exclusions.apply(instance_ids)       # 过滤后
    print(exclusions.summary())                 # 人类可读的说明
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import paths

# 与生成脚本共用同一组检测式。改这里**必须**重新生成清单，
# 否则清单与实际规则脱节 —— 所以带一个 rule_fingerprint 校验。
BAD_PATTERNS: list[tuple[str, str]] = [
    (r"\\[a-zA-Z]+[a-zA-Z]", "LaTeX 命令后直接跟字母（丢空格，如 \\pir 应为 \\pi r）"),
    (r"\^\{?\\", "上标缺失或嵌套错误"),
    (r"[a-zA-Z]\\[a-zA-Z]", "字母后紧跟命令且无空格"),
    (r"\$\s*\$", "空数学环境"),
]

_CACHE: set[str] | None = None


def rule_fingerprint() -> str:
    """规则的指纹，用于校验清单是不是用当前规则生成的。"""
    import hashlib
    blob = "|".join(p for p, _ in BAD_PATTERNS)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def list_path() -> Path:
    return paths.ROOT / "configs" / "exclusions.json"


def detect_bad_symbols(text: str) -> list[str]:
    """返回命中的模式描述；空列表表示干净。"""
    return [desc for pat, desc in BAD_PATTERNS if re.search(pat, text)]


def load(*, strict: bool = True) -> set[str]:
    """读排除清单。

    `strict=True` 会在**规则指纹不匹配**时报错而不是静默用旧清单 ——
    静默用错的清单会让"我剔除了缺陷题"这句话变成假的。
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = list_path()
    if not p.exists():
        _CACHE = set()
        return _CACHE
    d = json.loads(p.read_text(encoding="utf-8"))
    fp = d.get("rule_fingerprint")
    if strict and fp and fp != rule_fingerprint():
        raise RuntimeError(
            f"排除清单的规则指纹不匹配（清单={fp} 代码={rule_fingerprint()}）。\n"
            f"  清单是用**另一套规则**生成的。请重跑：\n"
            f"    .venv\\Scripts\\python.exe scripts/build_exclusions.py"
        )
    _CACHE = set(d.get("instance_ids") or [])
    return _CACHE


def apply(ids) -> list:
    """过滤掉被排除的题。保持原顺序。"""
    drop = load()
    return [i for i in ids if i not in drop]


def is_excluded(instance_id: str) -> bool:
    return instance_id in load()


def summary() -> str:
    p = list_path()
    if not p.exists():
        return "排除清单：**不存在**（不剔除任何题）"
    d = json.loads(p.read_text(encoding="utf-8"))
    n = len(d.get("instance_ids") or [])
    return (f"排除清单：剔除 {n} 题（题集共 {d.get('pool_size', '?')} 题）"
            f"｜依据：{d.get('reason', '?')}｜生成于 {str(d.get('generated_at'))[:19]}")


def reset_cache() -> None:
    """测试用：清缓存。"""
    global _CACHE
    _CACHE = None
