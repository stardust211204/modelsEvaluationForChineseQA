"""截断偏倚区间的回归测试。

守的是一条**会让结论偏乐观**的口径：四档判分把 `truncated` 排除在准确率分母外。
截断不是随机的 —— 实测被截断的题系统性更难（同批五个模型 35.4% vs 未截断 60.4%），
所以剔除它们会**高估**准确率。截断率低时无所谓（五主模型 0–0.3%，偏倚 ≤0.1pp），
截断率高时必须给区间（space-bunny 13.5%，偏倚 +3.1pp）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from meval import metrics  # noqa: E402


def _t(iid, tier, arm="a", mode="strict"):
    return {"arm": arm, "mode": mode, "tier": tier, "instance_id": iid}


def test_no_truncation_gives_no_interval():
    """没有截断时不该编出一个区间 —— acc_upper 应为 None。"""
    ts = [_t(f"q{i}", "correct" if i < 8 else "wrong") for i in range(10)]
    r = metrics.truncation_bias(ts, "a", {f"q{i}": 0.5 for i in range(10)})
    assert r["truncated_rate"] == 0.0
    assert r["acc_upper"] is None and r["bias_pp"] is None
    assert r["acc_gradeable"] == pytest.approx(0.8)


def test_truncated_items_pull_accuracy_down():
    """★ 核心：截断项若属于难题，外推后的准确率必须**低于**只算可判项的值。"""
    # 10 题：8 题可判（全对），2 题截断；截断题的难度代理远低于可判题
    ts = [_t(f"q{i}", "correct") for i in range(8)]
    ts += [_t("q8", "truncated"), _t("q9", "truncated")]
    diff = {f"q{i}": 0.9 for i in range(8)}       # 可判题：容易
    diff.update({"q8": 0.1, "q9": 0.1})           # 截断题：很难
    r = metrics.truncation_bias(ts, "a", diff)
    assert r["acc_gradeable"] == pytest.approx(1.0)
    assert r["acc_upper"] is not None and r["acc_upper"] < 1.0
    assert r["bias_pp"] > 0, "剔掉难题会高估，bias_pp 必须为正"


def test_invalid_is_not_truncation():
    """`invalid` 与 `truncated` 是不同档位，不能混算。"""
    ts = [_t("q0", "invalid"), _t("q1", "truncated"), _t("q2", "correct")]
    r = metrics.truncation_bias(ts, "a", {"q0": 0.5, "q1": 0.5, "q2": 0.5})
    assert r["n_truncated"] == 1, "只应把 truncated 计入截断"
    assert r["n_gradeable"] == 1


def test_item_difficulty_ignores_non_binary_tiers():
    """难度代理只用 correct/wrong；invalid/truncated 不代表能力，计进去会失真。"""
    ts = ([_t("q0", "correct", arm=a) for a in "abc"]
          + [_t("q1", "wrong", arm=a) for a in "abc"]
          + [_t("q2", "correct", arm=a) for a in "abc"]
          + [_t("q2", "invalid", arm=a) for a in "abc"])   # 干扰项，应被忽略
    d = metrics.item_difficulty(ts, min_arms=3)
    assert d["q0"] == pytest.approx(1.0)
    assert d["q1"] == pytest.approx(0.0)
    assert d["q2"] == pytest.approx(1.0), "invalid 不该被当成答错"


def test_min_arms_filter():
    """不足 min_arms 的题不产生难度值（样本太少的"难度"没有意义）。"""
    ts = [_t("q0", "correct", arm="a"), _t("q0", "wrong", arm="b")]
    assert metrics.item_difficulty(ts, min_arms=3) == {}
    assert "q0" in metrics.item_difficulty(ts, min_arms=2)
