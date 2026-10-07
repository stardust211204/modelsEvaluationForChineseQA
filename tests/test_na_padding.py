"""N/A 填充过滤的回归测试。

背景
    MMLU-ProX-Lite 用 "N/A" 字符串填充不足的选项位。
    `[o for o in options if o]` 只过滤 None / 空串，"N/A" 会漏过去，
    被原样渲染进提示词 —— validation 切片 70 题里 21 题有这问题。

这些不变量一旦被破坏：
  * 过滤掉「含 N/A 的题」⇒ 示例池只剩 10 选项题，反而加剧与 C-Eval 的错配
  * 中间位置的 N/A 被静默删除 ⇒ gold_indices 错位，判分全错
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from meval import build_set, paths  # noqa: E402


def test_strip_trailing_na():
    assert build_set._strip_na_options(["a", "b", "c", "d", "N/A", "N/A"]) == ["a", "b", "c", "d"]


def test_strip_noop_when_clean():
    opts = ["a", "b", "c"]
    assert build_set._strip_na_options(opts) == opts


@pytest.mark.parametrize("tok", ["N/A", "n/a", "NA", "N.A."])
def test_recognises_token_variants(tok):
    assert build_set._strip_na_options(["a", "b", tok]) == ["a", "b"]


def test_middle_na_raises_not_silently_drops():
    """★ 中间位置的 N/A 必须**报错** —— 静默删除会让 gold 下标错位。"""
    with pytest.raises(ValueError, match="中间位置"):
        build_set._strip_na_options(["a", "N/A", "c"])


def test_all_na_is_not_treated_as_trailing():
    """全 N/A 时 first=0，尾随判定成立 ⇒ 返回空列表（调用方需自行处理）。"""
    assert build_set._strip_na_options(["N/A", "N/A"]) == []


def test_frozen_sets_have_no_na():
    """★ 两个主评测集必须完全无 N/A —— 有的话说明构建流程漏了过滤。"""
    for name in ("eval_set", "ceval_set"):
        p = paths.data() / f"{name}.jsonl"
        if not p.exists():
            pytest.skip(f"{name} 不存在")
        bad = [i.id for i in build_set.load_frozen(p)
               if any(o.strip().upper() in build_set._NA_TOKENS for o in i.options)]
        assert not bad, f"{name} 有 {len(bad)} 题带 N/A 填充：{bad[:5]}"


def test_valid_pool_na_all_trailing():
    """★ 示例池若还有 N/A，必须全部尾随且答案在有效区内。"""
    p = paths.data() / "valid_pool.jsonl"
    if not p.exists():
        pytest.skip("valid_pool 不存在")
    offenders = []
    for i in build_set.load_frozen(p):
        flags = [o.strip().upper() in build_set._NA_TOKENS for o in i.options]
        if not any(flags):
            continue
        first = flags.index(True)
        if not all(flags[first:]) or any(g >= first for g in i.gold_indices):
            offenders.append(i.id)
    assert not offenders, (
        f"{len(offenders)} 题的 N/A 不在尾部或答案落在填充区：{offenders[:5]}\n"
        f"  删除这类填充会错位，必须重映射 gold_indices。"
    )
