"""配对比较单测。

这一层是报告里"Δpp 与显著性"的直接来源，算错不会被任何其它检查发现
（数字看起来永远合理），所以必须自己钉住：
  * McNemar 精确 p 与手算值一致
  * 并列的重复采样**不参与**配对（不得硬选一个制造信息量）
  * 一侧无法二元化时必须报告"不可配对"，而不是报一个 Δpp=0 之类的假差异
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from meval.metrics import mcnemar_exact, paired_compare, reduce_reps  # noqa: E402
from meval.types import ParseMode  # noqa: E402


# --------------------------------------------------------------------------
# McNemar 精确检验
# --------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs_is_nan():
    """两侧结论完全一致 -> 没有信息 -> nan（不是 p=1，也不是 p=0）。"""
    assert math.isnan(mcnemar_exact(0, 0))


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(4, 27) == pytest.approx(mcnemar_exact(27, 4))


def test_mcnemar_matches_hand_computation():
    """b=1,c=9 -> 2*(C(10,0)+C(10,1))/2^10 = 22/1024 = 0.021484375。"""
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)


def test_mcnemar_matches_reported_c2_value():
    """C1→C2 deepseek 的实测值：b=4, c=27 -> 3.40e-05（报告里的那个数）。"""
    p = mcnemar_exact(4, 27)
    assert p == pytest.approx(3.40e-05, abs=1e-6)


def test_mcnemar_never_exceeds_one():
    for b, c in ((0, 1), (1, 1), (2, 3), (5, 6), (0, 50)):
        assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_mcnemar_more_imbalance_is_more_significant():
    assert mcnemar_exact(10, 10) > mcnemar_exact(1, 19)


# --------------------------------------------------------------------------
# 重复采样压缩
# --------------------------------------------------------------------------

def test_reduce_reps_single_passthrough():
    assert reduce_reps(["correct"]) == "correct"


def test_reduce_reps_majority():
    assert reduce_reps(["correct", "correct", "wrong"]) == "correct"


def test_reduce_reps_tie_is_none():
    """2:2 并列时返回 None —— 硬选一个会凭空制造一个"答案"。"""
    assert reduce_reps(["correct", "correct", "wrong", "wrong"]) is None


def test_reduce_reps_empty_is_none():
    assert reduce_reps([]) is None


# --------------------------------------------------------------------------
# paired_compare
# --------------------------------------------------------------------------

def _t(iid: str, tier: str, cfg: str, arm="armA", mode="strict", rep=0) -> dict:
    return {"instance_id": iid, "arm": arm, "config": cfg, "rep": rep,
            "mode": mode, "tier": tier, "answer_index": 0,
            "tier_step": "bare", "invalid_reason": "", "attribution": None}


def test_paired_compare_basic_delta_and_counts():
    """4 题：1 题改善、1 题退步、2 题相同 -> b=1,c=1，Δpp=0，不显著。"""
    tallies = [
        _t("q1", "wrong", "c1"),   _t("q1", "correct", "c2"),   # 改善 -> c
        _t("q2", "correct", "c1"), _t("q2", "wrong", "c2"),     # 退步 -> b
        _t("q3", "correct", "c1"), _t("q3", "correct", "c2"),
        _t("q4", "wrong", "c1"),   _t("q4", "wrong", "c2"),
    ]
    r = paired_compare(tallies, base_arm="armA", other_arm="armA",
                       base_config="c1", other_config="c2")
    assert r["n_shared"] == 4
    assert r["n_paired"] == 4
    assert r["base_correct_other_not"] == 1
    assert r["base_not_other_correct"] == 1
    assert r["delta_pp"] == pytest.approx(0.0)
    # 两边各 2/4 对 -> 准确率都是 50%
    assert r["acc_base"] == pytest.approx(0.5)
    assert r["acc_other"] == pytest.approx(0.5)


def test_paired_compare_only_uses_shared_instances():
    """基准里多出来的题必须被排除 —— 这正是"不能直接相减"的原因。"""
    tallies = [
        _t("q1", "correct", "c1"), _t("q1", "wrong", "c2"),
        # 只在基准里出现的题：若被计入，基准准确率会被拉高
        _t("only_c1_a", "correct", "c1"), _t("only_c1_b", "correct", "c1"),
    ]
    r = paired_compare(tallies, base_arm="armA", other_arm="armA",
                       base_config="c1", other_config="c2")
    assert r["n_shared"] == 1
    assert r["acc_base"] == pytest.approx(1.0)   # 1/1，不是 3/3
    assert r["acc_other"] == pytest.approx(0.0)


def test_paired_compare_offscale_excluded_and_reported():
    """对照侧全是 invalid（严格口径遇上自由作答）-> 可配对 0，覆盖率 0。

    这时必须报"不可配对"，绝不能报一个 Δpp=0 —— 那会被读成"没有差异"。
    """
    tallies = [
        _t("q1", "correct", "c1"), _t("q1", "invalid", "c2"),
        _t("q2", "wrong", "c1"),   _t("q2", "truncated", "c2"),
    ]
    r = paired_compare(tallies, base_arm="armA", other_arm="armA",
                       base_config="c1", other_config="c2")
    assert r["n_shared"] == 2
    assert r["n_paired"] == 0
    assert r["n_offscale"] == 2
    assert r["coverage"] == 0.0
    assert r["mcnemar_p"] is None


def test_paired_compare_tie_reps_excluded():
    """重复采样并列的题不参与配对，且如实计入 n_tie_reps。"""
    tallies = [
        _t("q1", "correct", "c1"),
        _t("q1", "correct", "c2"), _t("q1", "wrong", "c2"),   # 2:2 -> 不可能，但 1:1 并列
        _t("q2", "correct", "c1"), _t("q2", "correct", "c2"),
    ]
    r = paired_compare(tallies, base_arm="armA", other_arm="armA",
                       base_config="c1", other_config="c2")
    assert r["n_shared"] == 2
    assert r["n_tie_reps"] == 1
    assert r["n_paired"] == 1


def test_paired_compare_no_shared_instances():
    tallies = [_t("q1", "correct", "c1"), _t("q2", "correct", "c2")]
    r = paired_compare(tallies, base_arm="armA", other_arm="armA",
                       base_config="c1", other_config="c2")
    assert r["n_shared"] == 0


def test_paired_compare_respects_mode():
    """严格口径下 C2 型作答无二元判定；宽松口径下才有 —— 口径必须隔离。"""
    tallies = [
        _t("q1", "correct", "c1", mode="strict"), _t("q1", "invalid", "c2", mode="strict"),
        _t("q1", "correct", "c1", mode="lenient"), _t("q1", "correct", "c2", mode="lenient"),
    ]
    strict = paired_compare(tallies, "armA", "armA", "c1", "c2", mode=ParseMode.STRICT)
    lenient = paired_compare(tallies, "armA", "armA", "c1", "c2", mode=ParseMode.LENIENT)
    assert strict["n_paired"] == 0
    assert lenient["n_paired"] == 1
