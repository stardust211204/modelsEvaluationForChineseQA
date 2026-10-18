"""回归门禁单测。

门禁本身如果坏了，比没有门禁更危险 —— 它会给人"已经检查过了"的错觉。
所以这里既测它的辅助函数，也测**核心断言确实能失败**（不然就是永真的假门禁）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import pytest  # noqa: E402

import ci_gate  # noqa: E402
from meval import adapt, metrics  # noqa: E402
from meval.types import Config, Instance, Source, Split  # noqa: E402


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------

def test_norm_is_key_order_insensitive():
    assert ci_gate.norm({"a": 1, "b": 2}) == ci_gate.norm({"b": 2, "a": 1})


def test_norm_detects_value_change():
    assert ci_gate.norm({"a": 1}) != ci_gate.norm({"a": 2})


def test_is_smoke_marks_rehearsals():
    assert ci_gate.is_smoke("pipe-smoke")
    assert ci_gate.is_smoke("c1-pilot")
    assert not ci_gate.is_smoke("c1-full")


def test_sha16_is_stable_and_short():
    assert ci_gate.sha16(b"abc") == ci_gate.sha16(b"abc")
    assert len(ci_gate.sha16(b"abc")) == 16


# --------------------------------------------------------------------------
# expected_specs：必须与跑批同一套抽样规则
# --------------------------------------------------------------------------

def _inst(i: int) -> Instance:
    return Instance(id=f"q{i}", question=f"题 {i}", options=["a", "b", "c"],
                    gold_indices=[0], split=Split.TEST,
                    stratum={"subject": "math", "difficulty": "unknown"},
                    source=Source.PUBLIC)


def _spec(key: Config, n: int, reps: int = 1) -> adapt.ConfigSpec:
    return adapt.ConfigSpec(key=key, name="t", factor=None, desc="",
                            prompt_template="prompts/c1_strict.txt",
                            n_instances=n, reps=reps)


def test_expected_specs_c1_uses_whole_pool():
    pool = [_inst(i) for i in range(10)]
    # C1 用全量，不看 n_instances
    assert ci_gate.expected_specs(pool, _spec(Config.C1, 600), n_arms=3) == 30


def test_expected_specs_non_c1_is_subset_times_arms_times_reps():
    pool = [_inst(i) for i in range(588)]
    # C3: 100 题 × 3 个模型 × 10 次
    assert ci_gate.expected_specs(pool, _spec(Config.C3, 100, reps=10), n_arms=3) == 3000
    # C2: 150 题 × 3 个模型 × 1 次
    assert ci_gate.expected_specs(pool, _spec(Config.C2, 150), n_arms=3) == 450


def test_expected_specs_subset_matches_adapt():
    """门禁的期望值不能自己另算一套 —— 必须与 adapt.subset_from_c1 一致。"""
    pool = [_inst(i) for i in range(588)]
    spec = _spec(Config.C2, 150)
    assert len(adapt.subset_from_c1(pool, 150)) == 150
    assert ci_gate.expected_specs(pool, spec, n_arms=3) == 3 * 150


# --------------------------------------------------------------------------
# prompt 指纹
# --------------------------------------------------------------------------

def test_prompt_fingerprint_is_deterministic(tmp_path):
    tmpl = tmp_path / "t.txt"
    tmpl.write_text("问：{question}\n{options}\n", encoding="utf-8")
    pool = [_inst(i) for i in range(3)]
    spec = _spec(Config.C1, 3)
    spec = adapt.ConfigSpec(key=spec.key, name=spec.name, factor=None, desc="",
                            prompt_template=str(tmpl), n_instances=3)
    a = ci_gate.prompt_fingerprint(pool, spec)
    b = ci_gate.prompt_fingerprint(pool, spec)
    assert a == b


def test_prompt_fingerprint_changes_when_template_changes(tmp_path):
    """模板改一个字，指纹必须变 —— 这正是"跨 run 比较失效"要被抓到的地方。"""
    tmpl = tmp_path / "t.txt"
    tmpl.write_text("问：{question}\n{options}\n", encoding="utf-8")
    pool = [_inst(i) for i in range(3)]
    spec = adapt.ConfigSpec(key=Config.C1, name="t", factor=None, desc="",
                            prompt_template=str(tmpl), n_instances=3)
    before = ci_gate.prompt_fingerprint(pool, spec)
    tmpl.write_text("问：{question}\n{options}\n请只回答字母。\n", encoding="utf-8")
    assert ci_gate.prompt_fingerprint(pool, spec) != before


# --------------------------------------------------------------------------
# ★ 核心断言必须真的能失败（否则是永真门禁）
# --------------------------------------------------------------------------

def _tally(iid, tier, mode, arm="armA"):
    return {"instance_id": iid, "arm": arm, "config": "c1", "rep": 0,
            "mode": mode, "tier": tier, "answer_index": 0,
            "tier_step": "bare", "invalid_reason": "", "attribution": None}


def test_gate_invariant_passes_when_only_parsing_differs():
    """严格判 invalid、宽松判出答案 —— 这是解析覆盖差异，不是判分分歧，应通过。"""
    tallies = [
        _tally("q1", "correct", "strict"), _tally("q1", "correct", "lenient"),
        _tally("q2", "invalid", "strict"), _tally("q2", "wrong", "lenient"),
    ]
    b = metrics.judge_agreement_report(tallies, "armA")["binary"]
    assert b["confusion"]["FN"] == 0 and b["confusion"]["FP"] == 0


def test_gate_invariant_fails_on_real_verdict_disagreement():
    """同一份回答，严格判 correct 而宽松判 wrong -> FN>0 —— 门禁必须能抓到。"""
    tallies = [
        _tally("q1", "correct", "strict"), _tally("q1", "wrong", "lenient"),
    ]
    b = metrics.judge_agreement_report(tallies, "armA")["binary"]
    assert b["confusion"]["FN"] == 1
