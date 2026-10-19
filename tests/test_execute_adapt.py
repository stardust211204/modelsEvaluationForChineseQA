"""执行层与请求构造层单测。

重点覆盖**断点续跑**——它是最容易写错、且错了会静默重跑烧额度的部分。
以及 `adapt` 的纯函数性——它错了会导致"不可复现"这类最难查的问题。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from meval import adapt, execute, paths  # noqa: E402
from meval.types import Config, Instance, PromptSpec, Split  # noqa: E402


# --------------------------------------------------------------------------
# prompt 指纹 —— 防「改了模板却沿用同一 run_id 静默复用旧数据」
# --------------------------------------------------------------------------

def _inst(i: int) -> Instance:
    return Instance(id=f"q{i}", question=f"题 {i}", options=["a", "b"],
                    gold_indices=[0], split=Split.TEST,
                    stratum={"subject": "math"}, source=None)  # type: ignore[arg-type]


def test_prompt_fingerprint_is_deterministic():
    insts = [_inst(i) for i in range(5)]
    assert (adapt.prompt_fingerprint(insts, "问：{question}\n{options}\n")
            == adapt.prompt_fingerprint(insts, "问：{question}\n{options}\n"))


def test_prompt_fingerprint_changes_with_template():
    """★ 模板改一个字，指纹必须变 —— 这正是"续跑会静默复用旧数据"要拦的东西。"""
    insts = [_inst(i) for i in range(5)]
    a = adapt.prompt_fingerprint(insts, "问：{question}\n{options}\n")
    b = adapt.prompt_fingerprint(insts, "问：{question}\n{options}\n只回答字母。\n")
    assert a != b


def test_prompt_fingerprint_changes_with_demos():
    """few-shot 示例变了，指纹也要变（C5 的模板修复就属于这一类）。"""
    insts = [_inst(i) for i in range(5)]
    tmpl = "{demos}\n问：{question}\n{options}\n"
    none = adapt.prompt_fingerprint(insts, tmpl, None)
    with_demo = adapt.prompt_fingerprint(insts, tmpl, [_inst(99)])
    assert none != with_demo


def test_prompt_fingerprint_changes_with_instance_set():
    """换了题集切片也要变 —— 指纹覆盖的是"实际会发出去的那些 prompt"。"""
    a = adapt.prompt_fingerprint([_inst(i) for i in range(5)], "问：{question}")
    b = adapt.prompt_fingerprint([_inst(i) for i in range(6)], "问：{question}")
    assert a != b


# --------------------------------------------------------------------------
# 夹具：临时把 runs 根指到 tmp_path
# --------------------------------------------------------------------------

@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def write_raw(tmp_path: Path, run_id: str, config: str, arm: str, rows: list[dict]) -> Path:
    d = tmp_path / "runs" / run_id / "raw" / config
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{arm}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def row(config: str, arm: str, iid: str, ok: bool, rep: int = 0,
        target_arm: str = "") -> dict:
    return {
        "config": config, "arm": arm, "instance_id": iid, "rep": rep,
        "target_arm": target_arm,
        "success": ok, "content": "B" if ok else "", "error": "" if ok else "boom",
    }


# --------------------------------------------------------------------------
# 断点续跑
# --------------------------------------------------------------------------
# 去重键是 5 元组：(config, arm, instance_id, rep, target_arm)。
# target_arm 是 C4 judge 引入的 —— 见下面最后一条用例。

def test_scan_done_empty_when_no_runs(isolated_runs):
    assert execute.scan_done("nope") == set()


def test_scan_done_counts_successes(isolated_runs):
    write_raw(isolated_runs, "r1", "c1", "armA",
              [row("c1", "armA", "q1", True), row("c1", "armA", "q2", True)])
    done = execute.scan_done("r1")
    assert done == {("c1", "armA", "q1", 0, ""), ("c1", "armA", "q2", 0, "")}


def test_scan_done_excludes_failures(isolated_runs):
    """失败的条目不算已完成 —— 否则重跑时会被永久跳过。"""
    write_raw(isolated_runs, "r1", "c1", "armA",
              [row("c1", "armA", "q1", True), row("c1", "armA", "q2", False)])
    done = execute.scan_done("r1")
    assert ("c1", "armA", "q1", 0, "") in done
    assert ("c1", "armA", "q2", 0, "") not in done


def test_scan_done_later_success_overrides_earlier_failure(isolated_runs):
    """先失败后重试成功 -> 算完成，不应再被重复调用。"""
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", False)])
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    assert ("c1", "armA", "q1", 0, "") in execute.scan_done("r1")


def test_scan_done_later_failure_overrides_earlier_success(isolated_runs):
    """顺序相反时以最后一次为准（不因历史成功而跳过）。"""
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", False)])
    assert ("c1", "armA", "q1", 0, "") not in execute.scan_done("r1")


def test_scan_done_ignores_partial_line(isolated_runs):
    """崩溃残留的半截行必须被忽略，不能让整个续跑失败。"""
    p = write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    with p.open("a", encoding="utf-8") as f:
        f.write('{"config": "c1", "arm": "armA", "instance')  # 半截
    assert execute.scan_done("r1") == {("c1", "armA", "q1", 0, "")}


def test_scan_done_distinguishes_reps(isolated_runs):
    """C3 的重复采样：同一题不同 rep 必须视为不同键。"""
    write_raw(isolated_runs, "r1", "c3", "armA",
              [row("c3", "armA", "q1", True, rep=r) for r in range(3)])
    done = execute.scan_done("r1")
    assert done == {("c3", "armA", "q1", r, "") for r in range(3)}


def test_scan_done_distinguishes_target_arm(isolated_runs):
    """★ C4 judge：同一 judge 模型对三个模型的**同一题**各判一次。

    三次调用的 (config, arm, instance_id, rep) 完全相同（arm 都是 judge 模型），
    只有 target_arm 不同。键里不带它就会三个模型互相覆盖、只剩最后一个模型
    —— 「结果字典键必须含 arm」这个坑在本项目里踩过两次。
    """
    write_raw(isolated_runs, "r1", "c4", "judge",
              [row("c4", "judge", "q1", True, target_arm=a)
               for a in ("armA", "armB", "armC")])
    done = execute.scan_done("r1")
    assert done == {("c4", "judge", "q1", 0, a) for a in ("armA", "armB", "armC")}
    assert len(done) == 3, "三个模型的 judge 请求必须是三个不同的键，不能塌成一个"


def test_run_specs_skips_everything_when_done(isolated_runs):
    """全部已完成时不该发起任何调用。"""
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    specs = [PromptSpec(run_id="r1", config=Config.C1, arm="armA", instance_id="q1")]

    class Boom:
        def call(self, spec):  # pragma: no cover - 不该被调用
            raise AssertionError("不应发起调用")

    stats = execute.run_specs(
        run_id="r1", specs=specs, arms_concurrency={"armA": 1},
        client=Boom(), progress=False,
    )
    assert stats["executed"] == 0
    assert stats["skipped_done"] == 1


# --------------------------------------------------------------------------
# load_results
# --------------------------------------------------------------------------

def test_load_results_dedups_by_key(isolated_runs):
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", False)])
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    got = execute.load_results("r1")
    assert len(got) == 1
    assert got[0].success is True


def test_load_results_filters_config(isolated_runs):
    write_raw(isolated_runs, "r1", "c1", "armA", [row("c1", "armA", "q1", True)])
    write_raw(isolated_runs, "r1", "c2", "armA", [row("c2", "armA", "q1", True)])
    assert len(execute.load_results("r1", Config.C1)) == 1
    assert len(execute.load_results("r1")) == 2


# --------------------------------------------------------------------------
# adapt：渲染与纯函数性
# --------------------------------------------------------------------------

@pytest.fixture
def inst() -> Instance:
    return Instance(
        id="q1", question="题干", options=["甲", "乙", "丙", "丁"], gold_indices=[1],
    )


def test_render_options_letters(inst):
    assert adapt.render_options(inst) == "(A) 甲\n(B) 乙\n(C) 丙\n(D) 丁"


def test_render_prompt_fills_placeholders(inst):
    out = adapt.render_prompt(inst, "题目：{question}\n\n{options}\n\n答案：")
    assert "题干" in out and "(C) 丙" in out


def test_render_prompt_is_pure(inst):
    tmpl = "题目：{question}\n{options}"
    assert adapt.render_prompt(inst, tmpl) == adapt.render_prompt(inst, tmpl)


def test_build_specs_is_pure(inst, tmp_path):
    """同一输入两次调用必须产出完全相同的 spec 列表。"""
    cfg = adapt.ConfigSpec(
        key=Config.C1, name="t", factor=None, desc="", prompt_template="x",
        n_instances=1,
    )
    arm = adapt.Arm(name="a", vendor="v", model="m", params={"max_tokens": 16})
    kw = dict(run_id="r", config=cfg, arms=[arm], instances=[inst],
              prompt_template="题目：{question}\n{options}")
    a = adapt.build_specs(**kw)
    b = adapt.build_specs(**kw)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_build_specs_reps(inst):
    """reps=10 应产生 10 条同题不同 rep 的 spec。"""
    cfg = adapt.ConfigSpec(
        key=Config.C3, name="t", factor=None, desc="", prompt_template="x",
        n_instances=1, reps=10,
    )
    arm = adapt.Arm(name="a", vendor="v", model="m")
    specs = adapt.build_specs(run_id="r", config=cfg, arms=[arm], instances=[inst],
                              prompt_template="{question}")
    assert len(specs) == 10
    assert {s.rep for s in specs} == set(range(10))
    assert len({s.key for s in specs}) == 10  # 键互不相同


def test_params_override(inst):
    cfg = adapt.ConfigSpec(
        key=Config.C2, name="t", factor=None, desc="", prompt_template="x",
        n_instances=1, params_override={"temperature": 0.7},
    )
    arm = adapt.Arm(name="a", vendor="v", model="m", params={"temperature": 0.1, "max_tokens": 9})
    specs = adapt.build_specs(run_id="r", config=cfg, arms=[arm], instances=[inst],
                              prompt_template="{question}")
    assert specs[0].params["temperature"] == 0.7   # override 生效
    assert specs[0].params["max_tokens"] == 9      # 其余保留


# --------------------------------------------------------------------------
# 子集抽样：C2 / C3 / C5 必须取自 C1
# --------------------------------------------------------------------------

def test_subset_is_real_subset():
    pool = [Instance(id=f"q{i}", question="t", options=["a", "b"], gold_indices=[0])
            for i in range(600)]
    sub = adapt.subset_from_c1(pool, 150)
    assert len(sub) == 150
    assert {i.id for i in sub} <= {i.id for i in pool}


def test_subset_is_deterministic():
    pool = [Instance(id=f"q{i}", question="t", options=["a", "b"], gold_indices=[0])
            for i in range(600)]
    assert [i.id for i in adapt.subset_from_c1(pool, 150)] == \
           [i.id for i in adapt.subset_from_c1(pool, 150)]


def test_subset_spans_whole_range():
    """等间隔取样应跨越整个题集，而不是集中在头部。"""
    pool = [Instance(id=f"q{i}", question="t", options=["a", "b"], gold_indices=[0])
            for i in range(600)]
    idxs = [int(i.id[1:]) for i in adapt.subset_from_c1(pool, 150)]
    assert max(idxs) > 500 and min(idxs) < 50


# --------------------------------------------------------------------------
# few-shot：绝不允许从 test split 取
# --------------------------------------------------------------------------

def test_load_demos_rejects_empty_valid_pool():
    c5 = adapt.ConfigSpec(
        key=Config.C5, name="t", factor=None, desc="", prompt_template="x",
        n_instances=1, few_shot={"n": 5},
    )
    test_only = [Instance(id="t1", question="q", options=["a"], gold_indices=[0],
                          split=Split.TEST)]
    with pytest.raises(RuntimeError, match="validation"):
        adapt.load_demos(c5, test_only)


def test_load_demos_picks_from_valid_only():
    c5 = adapt.ConfigSpec(
        key=Config.C5, name="t", factor=None, desc="", prompt_template="x",
        n_instances=1, few_shot={"n": 2},
    )
    pool = [
        Instance(id="t1", question="q", options=["a"], gold_indices=[0], split=Split.TEST),
        Instance(id="v1", question="q", options=["a"], gold_indices=[0], split=Split.VALID),
        Instance(id="v2", question="q", options=["a"], gold_indices=[0], split=Split.VALID),
    ]
    demos = adapt.load_demos(c5, pool)
    assert {d.id for d in demos} == {"v1", "v2"}
