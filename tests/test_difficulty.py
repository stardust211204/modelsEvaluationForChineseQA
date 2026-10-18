"""留一法共识难度单测。

这个脚本的价值全在一条：**它把"循环论证"从争论变成了一个可打印的数字**。
所以测试要钉住两件事：
  ① 标签规则本身（全对=easy / 全错=hard / 混合=mid / 有弃权=unknown）
  ② 全量共识在数学上必然退化成 100%/0%（恒等式），这就是它不能用的理由
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from difficulty_loo import consensus_label  # noqa: E402


def test_all_correct_is_easy():
    assert consensus_label(["correct", "correct"]) == "easy"


def test_all_wrong_is_hard():
    assert consensus_label(["wrong", "wrong", "wrong"]) == "hard"


def test_mixed_is_mid():
    assert consensus_label(["correct", "wrong"]) == "mid"
    assert consensus_label(["correct", "correct", "wrong"]) == "mid"


def test_single_arm_all_correct_is_easy():
    """单个模型也算全对（留一法下另一个模型缺席时不该崩）。"""
    assert consensus_label(["correct"]) == "easy"


def test_abstention_makes_it_unknown():
    """★ 有一个模型没给出二元判定 -> unknown。

    "没人答上来"和"很难"是两件事：invalid/truncated 归因于格式或我们的配置，
    混进难度会把配置问题记成题目难度。
    """
    assert consensus_label(["correct", "invalid"]) is None
    assert consensus_label(["wrong", "truncated"]) is None


def test_empty_is_unknown():
    assert consensus_label([]) is None
    assert consensus_label([""]) is None


def test_circular_consensus_is_a_tautology():
    """★ 全量共识（含目标模型）必然退化：标签来自目标模型自己的对错。

    三个模型全对 -> 目标模型必对（100%）；三个模型全错 -> 目标模型必错（0%）。
    这正是 HANDOFF §7.4「多模型共识」建议不能用 numerical 理由。
    """
    arms_tiers = ["correct", "correct", "correct"]
    assert consensus_label(arms_tiers) == "easy"
    # 目标模型是其中任意一个，其 tier 都必然是 correct -> 该桶准确率恒为 100%
    for t in arms_tiers:
        assert t == "correct"

    arms_tiers = ["wrong", "wrong", "wrong"]
    assert consensus_label(arms_tiers) == "hard"
    for t in arms_tiers:
        assert t == "wrong"
