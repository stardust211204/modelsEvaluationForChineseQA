"""重复采样（C3）一致性单测。

重点覆盖一处最容易被"看起来很确定"骗过去的逻辑：
    一道题 10 次全部 invalid，档位当然"全同" ——
    若把它计进一致率，等于把「永远不吐答案」当成「非常确定」。
因此断言必须把 n_answerable / n_never_answered 分开算。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from meval.metrics import consistency_c3, rep_accuracy_spread  # noqa: E402


def _t(iid: str, tier: str, rep: int, *, ans=None, mode="strict", arm="armA") -> dict:
    return {"instance_id": iid, "arm": arm, "config": "c3", "rep": rep,
            "mode": mode, "tier": tier, "answer_index": ans,
            "tier_step": "bare", "invalid_reason": "", "attribution": None}


# --------------------------------------------------------------------------
# consistency_c3
# --------------------------------------------------------------------------

def test_never_answered_is_not_counted_as_deterministic():
    """★ 核心：全 invalid 的题不能算进"确定"。

    q1 三次全 invalid（从不作答，档位却"全同"）
    q2 三次全 correct（答得出，且稳定）
    q3 三档 mixed（答得出，但不稳定）
    """
    tallies = [
        _t(i, "invalid", r) for i in ("q1",) for r in range(3)
    ] + [
        _t("q2", "correct", r, ans=0) for r in range(3)
    ] + [
        _t("q3", "correct", 0, ans=1), _t("q3", "wrong", 1, ans=2), _t("q3", "invalid", 2),
    ]
    c = consistency_c3(tallies, "armA")
    assert c["n_instances"] == 3
    assert c["n_never_answered"] == 1
    assert c["n_answerable"] == 2

    # 全量口径：q1（全 invalid）与 q2 都算"全同" -> 2/3，虚高
    assert c["tier_all_same_rate"] == pytest.approx(2 / 3, abs=1e-4)
    # 干净口径：只在可答题里算 -> 只有 q2 稳定 -> 1/2
    assert c["answerable_tier_all_same_rate"] == pytest.approx(0.5, abs=1e-4)
    assert c["answer_all_same_rate"] == pytest.approx(0.5, abs=1e-4)


def test_tier_stable_but_answer_flips():
    """档位全同不等于答案全同：全判 wrong 但选的选项每次都不同。

    这正是"只看档位会高估确定性"的具体形态。
    """
    tallies = [
        _t("q1", "wrong", 0, ans=1), _t("q1", "wrong", 1, ans=2), _t("q1", "wrong", 2, ans=3),
    ]
    c = consistency_c3(tallies, "armA")
    assert c["answerable_tier_all_same_rate"] == pytest.approx(1.0)
    assert c["answer_all_same_rate"] == pytest.approx(0.0)


def test_consistency_ignores_lenient_mode():
    """口径只取严格 —— 宽松口径的同一条 tally 不得让 rep 数翻倍。"""
    tallies = [_t("q1", "correct", 0, ans=0), _t("q1", "correct", 0, ans=0, mode="lenient")]
    c = consistency_c3(tallies, "armA")
    assert c["reps"] == pytest.approx(1.0)
    assert c["n_instances"] == 1


def test_consistency_filters_by_arm():
    tallies = [_t("q1", "correct", 0, arm="armA"), _t("q1", "wrong", 0, arm="armB")]
    assert consistency_c3(tallies, "armA")["n_instances"] == 1
    assert consistency_c3(tallies, "armZ") == {}


def test_consistency_empty():
    assert consistency_c3([], "armA") == {}


# --------------------------------------------------------------------------
# rep_accuracy_spread
# --------------------------------------------------------------------------

def test_rep_accuracy_spread_reports_range():
    """把每个 rep 当成一次独立缩量评测：极差就是这个点估计的重复噪声。"""
    tallies = [
        _t("q1", "correct", 0), _t("q2", "correct", 0),   # rep0 -> 2/2 = 1.00
        _t("q1", "correct", 1), _t("q2", "wrong", 1),     # rep1 -> 1/2 = 0.50
        _t("q1", "wrong", 2), _t("q2", "wrong", 2),       # rep2 -> 0/2 = 0.00
    ]
    s = rep_accuracy_spread(tallies, "armA")
    assert s["n_reps"] == 3
    assert s["n_per_rep"] == pytest.approx(2.0)
    assert s["acc_mean"] == pytest.approx(0.5, abs=1e-4)
    assert s["acc_min"] == 0.0 and s["acc_max"] == 1.0
    assert s["acc_range_pp"] == pytest.approx(100.0)


def test_rep_accuracy_excludes_invalid_from_denominator():
    """某个 rep 全是 invalid -> 该 rep 无定义，不应被算成 0% 拉低极差。"""
    tallies = [
        _t("q1", "correct", 0), _t("q2", "wrong", 0),
        _t("q1", "invalid", 1), _t("q2", "truncated", 1),
    ]
    s = rep_accuracy_spread(tallies, "armA")
    assert s["n_reps"] == 1
    assert s["acc_mean"] == pytest.approx(0.5, abs=1e-4)


def test_rep_accuracy_empty():
    assert rep_accuracy_spread([], "armA") == {}
