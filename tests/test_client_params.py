"""客户端参数分流单测。

针对一个**已实际踩到、且极难定位**的坑：
OpenAI SDK 的 chat.completions.create() 只接受它自己声明的参数名；
厂商私有参数（thinking / reasoning_effort…）直接当 kwarg 传，
会在**本地**抛 TypeError("unexpected keyword argument")，
根本到不了服务端 —— 而错误信息看起来像网络问题。

必须走 extra_body 透传。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meval.client import split_params  # noqa: E402


def test_standard_params_stay_as_kwargs():
    std, extra = split_params({"temperature": 0.1, "max_tokens": 512, "top_p": 0.9})
    assert std == {"temperature": 0.1, "max_tokens": 512, "top_p": 0.9}
    assert extra == {}


def test_vendor_params_go_to_extra_body():
    """★ 核心用例：这两个参数直传 kwarg 必抛 TypeError。"""
    std, extra = split_params({
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
    })
    assert std == {"temperature": 0.1}
    assert extra == {"thinking": {"type": "disabled"}, "reasoning_effort": "none"}


def test_empty_params():
    assert split_params({}) == ({}, {})
    assert split_params(None) == ({}, {})  # type: ignore[arg-type]


def test_seed_and_max_completion_tokens_are_standard():
    std, extra = split_params({"seed": 42, "max_completion_tokens": 100})
    assert std == {"seed": 42, "max_completion_tokens": 100}
    assert extra == {}


def test_classify_local_typeerror_is_not_retryable():
    """本地参数错误不该被当成可重试的网络问题（否则会白烧重试与额度）。"""
    from meval.client import classify
    from meval.types import ErrorKind

    kind, retryable = classify(TypeError("unexpected keyword argument 'thinking'"))
    assert kind is ErrorKind.CLIENT
    assert retryable is False


# --------------------------------------------------------------------------
# 重试成功必须清掉失败痕迹 ★
# --------------------------------------------------------------------------

def test_fill_clears_stale_error_from_previous_attempt():
    """★ 核心：先失败、退避重试后成功的记录，不得同时带着 error / error_kind。

    重试循环复用的是**同一个** ModelResult；异常分支会写 error / error_kind。
    成功分支若不清，就产出「success=True + http_status=200 + content 非空
    + 却带 transport 标签」的记录 —— 实测历史上有 60 / 5,769 条（1.04%）。

    success 判定本身没错（结论不受影响），但任何按 error 非空统计的分析
    都会得到假阳性，且与门禁的 by_error_kind 聚合口径冲突。
    """
    from types import SimpleNamespace

    from meval.client import ChatClient
    from meval.types import Config, ErrorKind, ModelResult

    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="答案：B", reasoning_content=""),
            finish_reason="stop",
        )],
        model="fake-model",
        usage=None,
    )

    res = ModelResult(instance_id="q1", arm="armA", config=Config.C1)
    # 模拟第一次尝试失败留下的痕迹
    res.error = "APIConnectionError: Connection error."
    res.error_kind = ErrorKind.TRANSPORT
    res.http_status = None

    ChatClient._fill(res, resp)

    assert res.success is True
    assert res.http_status == 200
    assert res.content == "答案：B"
    assert res.error == "", "成功后必须清空 error，否则下游按 error 统计会假阳性"
    assert res.error_kind is None


def test_fill_on_clean_result_still_has_no_error():
    """正常首次成功：error 保持空（防止把清理写成写入）。"""
    from types import SimpleNamespace

    from meval.client import ChatClient
    from meval.types import Config, ModelResult

    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="C", reasoning_content=None),
            finish_reason="stop",
        )],
        model="m",
        usage=None,
    )
    res = ModelResult(instance_id="q2", arm="armA", config=Config.C1)
    ChatClient._fill(res, resp)
    assert res.success is True
    assert res.error == ""
    assert res.error_kind is None
