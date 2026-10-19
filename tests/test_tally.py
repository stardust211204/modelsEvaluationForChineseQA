"""判分层单测。

这些用例是针对**已实际踩过的坑**固化的，不是为覆盖率写的：
  1. 空 content + finish_reason=length  -> truncated，不是 invalid（曾把整列答对误报成失败）
  2. 请求失败与格式失败必须分开
  3. 严格口径不能吃掉「答案其实在句子里」的输出（那是 C2 对照要量化的东西）
  4. 越界字母（4 选项却答 J）不算有效答案
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from meval.tally import parse_answer, tally_both_modes, tally_one  # noqa: E402
from meval.types import (  # noqa: E402
    Attribution,
    Config,
    ErrorKind,
    Instance,
    ModelResult,
    ParseMode,
    Tier,
)


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------

@pytest.fixture
def inst() -> Instance:
    return Instance(
        id="q1",
        question="测试题",
        options=["A项", "B项", "C项", "D项"],
        gold_indices=[2],  # 正确答案是 C
    )


def mk_result(
    content: str = "",
    *,
    reasoning: str = "",
    finish_reason: str = "stop",
    success: bool = True,
    error_kind: ErrorKind | None = None,
) -> ModelResult:
    return ModelResult(
        instance_id="q1",
        arm="test-arm",
        config=Config.C1,
        success=success,
        content=content,
        reasoning=reasoning,
        finish_reason=finish_reason,
        error_kind=error_kind,
    )


# --------------------------------------------------------------------------
# ★ 坑 1：空 content + length -> truncated（不是 invalid）
# --------------------------------------------------------------------------

def test_empty_content_with_length_is_truncated(inst):
    """推理模型把预算烧在思考上：HTTP 200，content 为空，finish_reason=length。

    这必须判为 truncated（我们配置的问题），而不是 invalid（模型的问题）。
    """
    res = mk_result("", reasoning="我们来看这道题……" * 100, finish_reason="length")
    t = tally_one(res, inst)
    assert t.tier is Tier.TRUNCATED
    assert t.tier is not Tier.INVALID


def test_empty_content_without_length_is_invalid(inst):
    """空输出但正常结束 -> 那是模型真的没答，算 invalid。"""
    res = mk_result("", finish_reason="stop")
    t = tally_one(res, inst)
    assert t.tier is Tier.INVALID
    assert t.tier is not Tier.TRUNCATED


def test_length_but_parsable_is_not_truncated(inst):
    """撞了长度上限，但答案已经给出来了 -> 照常判级，不算截断。

    判据保守化的意义：不要把「答对了但结尾被切」也算成我们的配置失败。
    """
    res = mk_result("C", finish_reason="length")
    t = tally_one(res, inst)
    assert t.tier is Tier.CORRECT


# --------------------------------------------------------------------------
# ★ 坑 2：请求失败 ≠ 格式失败
# --------------------------------------------------------------------------

def test_request_failure_is_invalid_with_reason(inst):
    res = mk_result("", success=False, error_kind=ErrorKind.QUOTA)
    t = tally_one(res, inst)
    assert t.tier is Tier.INVALID
    assert t.invalid_reason.startswith("request_failed")


# --------------------------------------------------------------------------
# 判级正确性
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "content,expected",
    [
        ("C", Tier.CORRECT),
        ("c", Tier.CORRECT),          # 小写
        ("(C)", Tier.CORRECT),        # 带括号
        ("C.", Tier.CORRECT),         # 带句点
        (" C ", Tier.CORRECT),        # 带空白
        ("B", Tier.WRONG),            # 选了错的
        ("", Tier.INVALID),           # 空
        ("我不知道", Tier.INVALID),    # 无法解析
    ],
)
def test_basic_grading(inst, content, expected):
    assert tally_one(mk_result(content), inst).tier is expected


def test_out_of_range_letter_is_invalid(inst):
    """4 个选项却答 J —— 不是有效答案，不能当成「选错了」。"""
    idx, step = parse_answer("J", n_options=4, mode=ParseMode.STRICT)
    assert idx is None
    assert step == "none"


def test_ten_option_question_accepts_j():
    """10 选项的题（MMLU-ProX 存在）必须接受 J。"""
    idx, _ = parse_answer("J", n_options=10, mode=ParseMode.STRICT)
    assert idx == 9


# --------------------------------------------------------------------------
# ★ 坑 3：严格 vs 宽松的口径差（C2 对照要量化的正是这个）
# --------------------------------------------------------------------------

def test_strict_rejects_but_lenient_accepts(inst):
    """输出是一句话。严格口径吃不下，宽松口径能抽出答案。

    这个差值就是「判分口径造成多少伪低分」——是本项目的核心观测量之一。
    """
    content = "经过分析，我认为答案是 C。"
    assert tally_one(mk_result(content), inst, ParseMode.STRICT).tier is Tier.INVALID
    assert tally_one(mk_result(content), inst, ParseMode.LENIENT).tier is Tier.CORRECT


def test_lenient_last_line_ladder(inst):
    """阶梯第三级：末行孤立字母。"""
    content = "推理过程……\nC"
    t = tally_one(mk_result(content), inst, ParseMode.LENIENT)
    assert t.tier is Tier.CORRECT
    assert t.tier_step == "last-line"


def test_phrase_forms(inst):
    for content in ["Answer: C", "answer is C", "答案是 C", "答案：C", "The answer is (C)"]:
        t = tally_one(mk_result(content), inst, ParseMode.LENIENT)
        assert t.tier is Tier.CORRECT, f"未识别: {content!r}"


# --------------------------------------------------------------------------
# ★ 坑 4：罗列选项的回答会被判成 A —— 一个**制造虚假竞品差异**的 bug
# --------------------------------------------------------------------------
# 实测背景（第 4 个模型 qwen 补跑时发现）：
#   原 PHRASE_RE 的触发词含 `选项`，且用 search 取**第一个**匹配。
#   于是"先列出所有选项、最后给答案"的回答必然命中第一个 `选项 (A)`：
#
#       输入：(A) $7.00 (B) $5.00 … 答案：J
#       旧行为：判成 A          ← 错
#       新行为：判成 J          ← 对
#
#   为什么这条 bug 特别危险：**它的影响按模型不均匀**。
#   话多、爱罗列选项的模型被系统性判成 A（实测 qwen 中位输出 739 token，
#   准确率被压到 48%），话少的模型几乎不受影响（deepseek 中位 123 token）。
#   ⇒ 解析器的缺陷伪装成了"模型能力差异"，而竞品分析正是要比模型能力。
#
#   这也解释了为什么原有 147 个用例全绿却没拦住它：
#   它们测的是"单个句式能否识别"，**没有一条测"多个候选同时出现时该选哪个"**。


def test_listed_options_do_not_override_final_answer(inst):
    """★ 核心回归：正文罗列了 (A)(B)(C)(D)，最终给的是 C —— 必须判 C。"""
    content = (
        "逐项分析：\n"
        "(A) $7.00 — 不符合\n"
        "(B) $5.00 — 不符合\n"
        "(C) $3.75 — 符合我们的计算\n"
        "(D) $2.50 — 不符合\n\n"
        "答案：C"
    )
    idx, step = parse_answer(content, n_options=10, mode=ParseMode.LENIENT)
    assert idx == 2, f"应判 C(idx=2)，实际判成 {idx}"
    assert step == "answer-marker"


def test_option_word_alone_is_not_an_answer_marker(inst):
    """「选项 (A) 是…」只是**在讨论选项**，不等于给出了答案。

    该句后面必须有明确的答案标记，否则不该据此定位答案。
    """
    content = "选项 (A) 的说法与题干不符，因为它混淆了两个概念。\n\n因此选 D。"
    idx, step = parse_answer(content, n_options=10, mode=ParseMode.LENIENT)
    assert idx == 3, f"应判 D(idx=3)，实际判成 {idx}"


def test_answer_marker_takes_the_last_occurrence(inst):
    """模型可能先猜后改 —— 以**最后一次**表态为准。

    A=0 B=1 C=2 D=3，所以正确答案 D 对应 idx=3。
    """
    content = "我先看成 A，但重算后发现不对。\n答案：A 是错的。\n正确答案：D"
    idx, _ = parse_answer(content, n_options=10, mode=ParseMode.LENIENT)
    assert idx == 3, f"应取最后一次表态 D(idx=3)，实际 {idx}"


def test_long_verbose_answer_still_parses(inst):
    """真实 qwen 输出的缩微版：长推理 + 选项罗列 + 结尾答案，10 选项。"""
    content = (
        "这是一道关于期望效用的题。\n\n"
        "**1. 确定初始状态**\n* 初始财富 = 20,000\n\n"
        "选项对比：\n"
        "选项 (A) 9.210\n选项 (B) 9.616\n选项 (C) 9.000\n"
        "选项 (D) 8.500\n选项 (E) 9.100\n选项 (F) 8.900\n"
        "选项 (G) 9.300\n选项 (H) 9.400\n选项 (I) 9.500\n选项 (J) 9.700\n\n"
        "计算结果与选项 (B) 吻合。\n\n答案：B"
    )
    idx, step = parse_answer(content, n_options=10, mode=ParseMode.LENIENT)
    assert idx == 1, f"应判 B(idx=1)，实际判成 {idx}"
    assert step == "answer-marker"


def test_tally_both_modes_returns_two(inst):
    out = tally_both_modes(mk_result("C"), inst)
    assert {t.mode for t in out} == {ParseMode.STRICT, ParseMode.LENIENT}
    assert all(t.tier is Tier.CORRECT for t in out)


# --------------------------------------------------------------------------
# 多正解（借 HELM 的 Reference.tags 机制）
# --------------------------------------------------------------------------

def test_multiple_gold_indices():
    """否定式提问可能有争议答案 —— 一题多正解不应被判错。"""
    inst = Instance(
        id="q9", question="以下哪个不是……", options=["甲", "乙", "丙", "丁"],
        gold_indices=[0, 3],
    )
    assert tally_one(mk_result("A"), inst).tier is Tier.CORRECT
    assert tally_one(mk_result("D"), inst).tier is Tier.CORRECT
    assert tally_one(mk_result("B"), inst).tier is Tier.WRONG


# --------------------------------------------------------------------------
# 归因
# --------------------------------------------------------------------------

def test_attribution_only_on_wrong(inst):
    from meval.tally import guess_attribution

    t = tally_one(mk_result("C"), inst)
    assert guess_attribution(mk_result("C"), inst, t) is None  # correct 档无归因


def test_attribution_adjacent_option_is_confusion(inst):
    from meval.tally import guess_attribution

    # gold = C(2)，选 B(1) 相邻
    res = mk_result("B")
    t = tally_one(res, inst)
    assert t.tier is Tier.WRONG
    assert guess_attribution(res, inst, t) is Attribution.OPTION_CONFUSION


def test_attribution_long_reasoning_is_reasoning(inst):
    from meval.tally import guess_attribution

    # gold = C(2)，选 A(0) 不相邻、思考很长
    res = mk_result("A", reasoning="x" * 500)
    t = tally_one(res, inst)
    assert guess_attribution(res, inst, t) is Attribution.REASONING


def test_attribution_falls_back_to_content_when_reasoning_empty(inst):
    """★ C1 关掉了思考（reasoning 恒空），C2 把推理写在 content 里。

    只看 `reasoning` 会让 C2 的归因全部退化成"事实错误" ——
    把"有推理但推错了"误判成"根本不知道"，方向性地扭曲归因分布。
    """
    from meval.tally import guess_attribution

    # gold = C(2)，选 A(0) 不相邻；reasoning 空但 content 很长
    # 用宽松口径解析（自由作答不是裸字母，严格口径会判 invalid）
    res = mk_result("答案是 A。" + "推理过程。" * 100)
    t = tally_one(res, inst, ParseMode.LENIENT)
    assert t.tier is Tier.WRONG
    assert guess_attribution(res, inst, t) is Attribution.REASONING


def test_attribution_short_content_stays_factual(inst):
    """短作答（没推理）仍然是事实错误，不能被误升级成推理问题。"""
    from meval.tally import guess_attribution

    res = mk_result("答案是 A")
    t = tally_one(res, inst, ParseMode.LENIENT)
    assert t.tier is Tier.WRONG
    assert guess_attribution(res, inst, t) is Attribution.FACTUAL
