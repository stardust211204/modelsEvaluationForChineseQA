"""指标层单测。

重点覆盖两处最容易出错的统计逻辑：
  1. bootstrap CI 必须**确定性**（固定 seed）—— 否则"可复现"是假的
  2. 四档分账必须**排除** invalid / truncated 出分母 ——
     否则会把"我 token 预算设小了"算成"模型能力不足"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from meval.metrics import (  # noqa: E402
    arm_to_row,
    bootstrap_ci,
    cohen_kappa,
    compute_arm,
    delta_between,
    format_acc,
    strict_lenient_agreement,
    tier_accounting,
    wilson_ci,
)
from meval.types import Config, ParseMode, Tier  # noqa: E402


# --------------------------------------------------------------------------
# bootstrap CI
# --------------------------------------------------------------------------

def test_bootstrap_ci_is_deterministic():
    """同一输入两次必须得到完全相同的区间 —— 固定 seed 的意义。"""
    hits = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 10
    a = bootstrap_ci(hits)
    b = bootstrap_ci(hits)
    assert a == b


def test_bootstrap_ci_contains_point():
    hits = [1] * 60 + [0] * 40
    point, lo, hi = bootstrap_ci(hits)
    assert lo <= point <= hi
    assert point == pytest.approx(0.6)


def test_bootstrap_ci_widens_when_uncertain():
    """方差大时区间应比方差小时更宽。"""
    tight = bootstrap_ci([1] * 95 + [0] * 5)
    loose = bootstrap_ci([1] * 50 + [0] * 50)
    assert (tight[2] - tight[1]) < (loose[2] - loose[1])


def test_bootstrap_ci_empty_and_single():
    p, lo, hi = bootstrap_ci([])
    assert math.isnan(p)
    assert bootstrap_ci([1]) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_matches_known_width():
    """600 题、p=0.5 时 95% CI 宽度应约 ±4pp（设计文档里的声明）。"""
    hits = [1] * 300 + [0] * 300
    _p, lo, hi = bootstrap_ci(hits)
    half_width_pp = (hi - lo) / 2 * 100
    assert 3.0 < half_width_pp < 5.0, f"半宽 {half_width_pp:.2f}pp 与声明的 ±4pp 不符"


# --------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------

def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0 < lo < 1
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def test_wilson_better_than_naive_at_extremes():
    """k=0 时正态近似会给出负下界；Wilson 不会。"""
    lo, _hi = wilson_ci(0, 20)
    assert lo >= 0.0


# --------------------------------------------------------------------------
# Kappa
# --------------------------------------------------------------------------

def test_kappa_perfect_agreement():
    a = ["correct", "wrong", "correct", "wrong"] * 10
    assert cohen_kappa(a, list(a)) == pytest.approx(1.0)


def test_kappa_chance_level_is_near_zero():
    """一个口径全判 correct、另一个随机 —— kappa 应接近 0。"""
    a = ["correct"] * 50 + ["correct"] * 50
    b = ["correct"] * 50 + ["wrong"] * 50
    k = cohen_kappa(a, b)
    assert math.isnan(k) or abs(k) < 0.05


def test_kappa_degenerate_returns_nan_not_crash():
    """两个口径都只输出同一类 -> 分母为 0 -> 返回 nan（如实报告，不硬造 0）。"""
    a = ["correct"] * 10
    b = ["correct"] * 10
    assert math.isnan(cohen_kappa(a, b))


def test_kappa_length_mismatch():
    assert math.isnan(cohen_kappa(["a"], []))


# --------------------------------------------------------------------------
# 四档分账（★ 核心）
# --------------------------------------------------------------------------

def _tally(iid, tier, *, mode="strict", arm="armA", step="bare", rep=0):
    return {"instance_id": iid, "arm": arm, "config": "c1", "rep": rep,
            "mode": mode, "tier": tier, "answer_index": None,
            "tier_step": step, "invalid_reason": "", "attribution": None}


@pytest.fixture
def insts():
    return {
        "q1": {"id": "q1", "stratum": {"subject": "math", "difficulty": "easy"}, "source": "public"},
        "q2": {"id": "q2", "stratum": {"subject": "math", "difficulty": "hard"}, "source": "public"},
        "q3": {"id": "q3", "stratum": {"subject": "law", "difficulty": "mid"}, "source": "public"},
    }


def test_accuracy_excludes_invalid_and_truncated(insts):
    """★ 关键：分母 = correct + wrong，invalid/truncated 不进分母。

    8 条里 4 对 2 错 1 invalid 1 truncated -> 准确率应为 4/6 = 66.7%，不是 4/8 = 50%。
    """
    tallies = [
        _tally("q1", "correct"), _tally("q1", "wrong"),
        _tally("q2", "correct"), _tally("q2", "wrong"),
        _tally("q3", "correct"),
        _tally("q1", "correct", rep=1),
        _tally("q2", "invalid"),
        _tally("q3", "truncated"),
    ]
    st = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                     results={}, instances=insts, mode=ParseMode.STRICT)
    assert st.n_total == 8
    assert st.n_correct == 4 and st.n_wrong == 2
    assert st.n_invalid == 1 and st.n_truncated == 1
    assert st.acc == pytest.approx(4 / 6)
    assert st.invalid_rate == pytest.approx(1 / 8)
    assert st.truncated_rate == pytest.approx(1 / 8)


def test_accuracy_nan_when_no_gradeable(insts):
    """全是 invalid -> 准确率无定义（nan），而不是 0。

    报成 0 会让"模型完全不行"和"我们的配置有问题"看起来一样。
    """
    tallies = [_tally("q1", "invalid"), _tally("q2", "truncated")]
    st = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                     results={}, instances=insts, mode=ParseMode.STRICT)
    assert math.isnan(st.acc)


def test_filters_by_mode_and_arm(insts):
    tallies = [
        _tally("q1", "correct", mode="strict", arm="armA"),
        _tally("q1", "wrong", mode="lenient", arm="armA"),
        _tally("q1", "correct", mode="strict", arm="armB"),
    ]
    a_strict = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                           results={}, instances=insts, mode=ParseMode.STRICT)
    a_len = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                        results={}, instances=insts, mode=ParseMode.LENIENT)
    assert a_strict.n_correct == 1 and a_strict.n_wrong == 0
    assert a_len.n_wrong == 1 and a_len.n_correct == 0


def test_stratum_breakdown(insts):
    tallies = [
        _tally("q1", "correct"), _tally("q1", "wrong"),   # math/easy
        _tally("q2", "wrong"),                            # math/hard
        _tally("q3", "correct"),                          # law/mid
    ]
    st = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                     results={}, instances=insts, mode=ParseMode.STRICT)
    math_cell = st.by_stratum["subject"]["math"]
    assert math_cell["n"] == 3 and math_cell["correct"] == 1
    assert math_cell["acc"] == pytest.approx(1 / 3, abs=1e-4)
    assert st.by_stratum["difficulty"]["easy"]["acc"] == pytest.approx(0.5, abs=1e-4)
    # by_source 现在只有一个来源（题库全来自公开集），所以这里验的是**聚合正确性**：
    # q1对 + q1错 + q2错 + q3对 = 2/4。（来源的**拆分**能力由上面的 by_stratum 覆盖。）
    assert st.by_source["public"]["acc"] == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------
# 口径一致率
# --------------------------------------------------------------------------

def test_strict_lenient_agreement_counts_rescued():
    """严格判 invalid、宽松判出答案 -> 这就是「判分口径造成的伪低分」。"""
    tallies = [
        _tally("q1", "correct", mode="strict"),
        _tally("q1", "correct", mode="lenient"),
        _tally("q2", "invalid", mode="strict"),
        _tally("q2", "correct", mode="lenient"),   # 被救回
        _tally("q3", "wrong", mode="strict"),
        _tally("q3", "wrong", mode="lenient"),
    ]
    r = strict_lenient_agreement(tallies, "armA")
    assert r["n"] == 3
    assert r["strict_invalid_lenient_parsed"] == 1
    assert r["agreement"] == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------
# Δacc 与导出
# --------------------------------------------------------------------------

def test_delta_between():
    from meval.metrics import ArmStats

    a = ArmStats(arm="x", config="c1", n_correct=70, n_wrong=30, acc=0.70)
    b = ArmStats(arm="x", config="c2", n_correct=60, n_wrong=40, acc=0.60)
    d = delta_between(a, b)
    assert d["delta_pp"] == pytest.approx(10.0)


def test_arm_to_row_has_ci_width():
    from meval.metrics import ArmStats

    st = ArmStats(arm="x", config="c1", n_total=100, n_correct=70, n_wrong=30,
                  acc=0.70, acc_lo=0.65, acc_hi=0.75,
                  invalid_rate=0.0, truncated_rate=0.0)
    row = arm_to_row(st)
    assert row["ci_width_pp"] == pytest.approx(10.0)
    assert row["acc"] == 0.70


# --------------------------------------------------------------------------
# 四档分账（唯一的准确率定义）★ 核心
# --------------------------------------------------------------------------

def test_tier_accounting_excludes_abstentions_from_denominator():
    """4 对 / 2 错 / 1 invalid / 1 truncated -> 准确率 4/6，不是 4/8。"""
    ac = tier_accounting(["correct"] * 4 + ["wrong"] * 2 + ["invalid", "truncated"])
    assert ac["n_total"] == 8
    assert ac["n_gradeable"] == 6
    assert ac["acc"] == pytest.approx(4 / 6)
    assert ac["invalid_rate"] == pytest.approx(1 / 8)
    assert ac["truncated_rate"] == pytest.approx(1 / 8)


def test_tier_accounting_none_when_nothing_gradeable():
    """★ 一条都判不出来 -> acc is None（无定义），**不是 0.0**。

    这是本项目最想避免的误读：把"不可判定"说成"全错"。
    真实案例：C2 的严格口径 410/450 判不出来，曾被打印成 0.0%。
    """
    ac = tier_accounting(["invalid"] * 410 + ["truncated"] * 40)
    assert ac["n_gradeable"] == 0
    assert ac["acc"] is None
    assert ac["invalid_rate"] == pytest.approx(410 / 450)


def test_tier_accounting_all_correct_is_one():
    ac = tier_accounting(["correct"] * 5)
    assert ac["acc"] == pytest.approx(1.0)


def test_tier_accounting_empty():
    ac = tier_accounting([])
    assert ac["n_total"] == 0
    assert ac["acc"] is None
    assert ac["invalid_rate"] is None


def test_tier_accounting_tiers_snapshot():
    ac = tier_accounting(["correct", "wrong", "wrong"])
    assert ac["tiers"] == {"correct": 1, "wrong": 2}
    assert ac["acc"] == pytest.approx(1 / 3)


def test_format_acc_never_renders_zero_for_undefined():
    """★ 无定义必须显示成「无定义」；0.0% 是合法的**真实**取值，不能混。"""
    assert format_acc(None) == "无定义"
    assert format_acc(0.0) == "0.0%"
    assert format_acc(0.7541) == "75.4%"


def test_tier_accounting_agrees_with_compute_arm(insts):
    """两套实现不许漂移：tier_accounting 的 acc 必须与 compute_arm 一致。"""
    tiers = ["correct", "wrong", "correct", "invalid", "truncated", "wrong"]
    ac = tier_accounting(tiers)
    tallies = [_tally(f"q{i}", t) for i, t in enumerate(tiers)]
    st = compute_arm(arm="armA", config=Config.C1, tallies=tallies,
                     results={}, instances=insts, mode=ParseMode.STRICT)
    assert st.acc == pytest.approx(ac["acc"])
    assert st.n_correct == ac["n_correct"]
    assert st.n_wrong == ac["n_wrong"]
