"""提示词的字母范围必须与**该题实际的选项数**一致。

这个测试守的是一类会造成**臂间不公平**的缺陷：
提示词写死「必须且只能是一个大写字母（A 到 J 之一）」，而 MMLU-ProX 有
103/588 题选项数 <10、C-Eval 583/583 题只有 4 个选项 —— 于是提示词
告诉模型"可以答 J"，选项里却没有 J。

实测（同题同模型，只改这一处，各 3 次）：

    提示词「A 到 J」 -> J, J, J     （不存在的选项 ⇒ 判 invalid）
    提示词「A 到 I」 -> I, I, I     （★ I 正是标准答案）

★ 为什么这类缺陷特别危险：**它按臂不均匀地生效**。严格照字面执行指令的臂
  （实测旗舰 2.4% 越界）被扣分，而会自己看选项列表的臂（四个 flash 臂 0–1.5%）
  不受影响 ⇒ 提示词的锅被记到了模型头上。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from meval import adapt, build_set, paths  # noqa: E402

PROMPT_DIR = _ROOT / "prompts"
# 会渲染进被测模型答题提示词的模板（judge 模板不适用这条规则）
ANSWER_TEMPLATES = ["c1_strict.txt", "c2_free.txt", "c5_fewshot.txt"]

RANGE_RE = re.compile(r"A\s*到\s*([A-J])")
ANY_RANGE_RE = re.compile(r"\bA\s*(?:到|-|~|～|to)\s*([A-Z])\b")


def _datasets():
    for name in ("eval_set", "ceval_set"):
        p = paths.data() / f"{name}.jsonl"
        if p.exists():
            yield name, build_set.load_frozen(p)


@pytest.mark.parametrize("tpl_name", ANSWER_TEMPLATES)
def test_rendered_prompt_never_mentions_nonexistent_letters(tpl_name):
    """★ 核心守卫：渲染后的提示词不得提到超出该题选项范围的字母。"""
    tpl_path = PROMPT_DIR / tpl_name
    if not tpl_path.exists():
        pytest.skip(f"{tpl_name} 不存在")
    tpl = tpl_path.read_text(encoding="utf-8")
    any_ds = False
    for ds_name, pool in _datasets():
        any_ds = True
        for inst in pool:
            rendered = adapt.render_prompt(inst, tpl)
            m = RANGE_RE.search(rendered)
            if not m:
                continue
            stated = m.group(1)
            actual_last = chr(64 + len(inst.options))
            assert stated == actual_last, (
                f"{tpl_name} 在 {ds_name}/{inst.id} 上渲染出『A 到 {stated}』，"
                f"但该题只有 {len(inst.options)} 个选项（A–{actual_last}）。\n"
                f"  提示词不能提到不存在的选项字母 —— 见本文件顶部的实测证据。"
            )
    if not any_ds:
        pytest.skip("没有可用题集")


def test_templates_do_not_hardcode_letter_range():
    """模板源文件里不得写死字母范围（必须用 {last} 占位符）。"""
    for tpl_name in ANSWER_TEMPLATES:
        p = PROMPT_DIR / tpl_name
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        # 「{last}」是我们要求的动态形式；除此之外不该出现写死的终字母
        hard = re.search(r"A\s*到\s*([A-J])(?!\})", src)
        assert not hard, (
            f"{tpl_name} 写死了『A 到 {hard.group(1)}』—— 必须改成『A 到 {{last}}』，\n"
            f"  否则选项数 <{ord(hard.group(1)) - 64} 的题会被误导。"
        )


@pytest.mark.parametrize("tpl_name", ANSWER_TEMPLATES)
def test_render_prompt_accepts_last_placeholder(tpl_name):
    """`{last}` 必须能被填上（占位符改名/漏传会抛 KeyError，这里提前拦住）。"""
    p = PROMPT_DIR / tpl_name
    if not p.exists():
        pytest.skip(f"{tpl_name} 不存在")
    tpl = p.read_text(encoding="utf-8")
    for _name, pool in _datasets():
        inst = pool[0]
        out = adapt.render_prompt(inst, tpl)
        assert "{last}" not in out, "占位符没被替换 —— render_prompt 漏了参数"
        return
    pytest.skip("没有可用题集")
