"""统一 OpenAI 兼容客户端。

职责（且仅此）：把一次 PromptSpec 发出去，把原始返回**原样**带回来。

设计要点：
  * 关闭 SDK 自动重试 —— 我们要自己区分 429 / 5xx / 超时 / 4xx
  * 同时读 `content` 与 `reasoning_content`
    —— 推理模型把思考写进独立字段，且思考与答案共享 token 预算；
       只读 content 会在思考吃满预算时拿到空串，而 HTTP 状态是 200（静默失败）
  * 记录 `finish_reason` 与 `reasoning_tokens`，供 tally 层识别截断
  * 记录 `request_datetime` —— API 背后的模型可能变化
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import httpx
import openai

from . import paths
from .types import Config, ErrorKind, ModelResult, PromptSpec

# --------------------------------------------------------------------------
# 端点适配
# --------------------------------------------------------------------------
# 聚合端点要求每个请求带会话标识，否则直接 400 MissingSessionID。
# 这是通用 OpenAI SDK 不会发的头，必须通过自定义 http_client 注入。
# 早期探针脚本（scripts/ 下的手写 urllib 版本）已经这么做了，这里是同一要求的正式实现。

_SESSION_HEADER = "X-Opencode-Session"
_EXTRA_HEADERS = {
    "User-Agent": "meval/0.1 (+reproducible-eval)",
}


def build_http_client(
    timeout_s: float = 120.0,
    *,
    base_url: str | None = None,
    trust_env: bool | None = None,
) -> httpx.Client:
    """构造带端点适配头的 httpx 客户端。

    ★ 关键：本地端点必须绕过系统代理。
      本机开着系统代理时（实测 127.0.0.1:7897），httpx 默认 trust_env=True
      会把**本地**请求也送进代理，代理对 localhost 返回 502 ——
      表现为"本地模型不可用"，但 Ollama 服务端日志里其实是 200。
      而 urllib 不读系统代理，所以直连正常，这个差异极难定位。
      远端聚合端点走代理没问题；本地端点一律 trust_env=False。
    """

    def _inject(request: httpx.Request) -> None:
        request.headers[_SESSION_HEADER] = str(uuid.uuid4())
        for k, v in _EXTRA_HEADERS.items():
            request.headers.setdefault(k, v)

    if trust_env is None:
        url = base_url if base_url is not None else paths.base_url()
        trust_env = not _is_local(url)

    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=15.0),
        event_hooks={"request": [_inject]},
        follow_redirects=True,
        trust_env=trust_env,
    )


def _is_local(url: str) -> bool:
    """URL 是否指向本机。"""
    return any(h in (url or "") for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))

# --------------------------------------------------------------------------
# 参数分流
# --------------------------------------------------------------------------
# OpenAI SDK 的 chat.completions.create() 只接受它自己声明的参数名。
# 厂商私有参数（thinking / reasoning_effort 之类）直接当 kwarg 传会在**本地**
# 抛 TypeError（"unexpected keyword argument"），根本到不了服务端。
# 这类参数必须走 extra_body 透传。
#
# 这个坑很隐蔽：错误看起来像网络问题，实际是本地参数校验失败。

_STANDARD_PARAMS = frozenset({
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "n",
    "stream", "stop", "presence_penalty", "frequency_penalty",
    "logit_bias", "user", "seed", "response_format", "tools", "tool_choice",
    "logprobs", "top_logprobs", "parallel_tool_calls",
})


def split_params(params: dict) -> tuple[dict, dict]:
    """把参数分成 (标准参数, 需要透传的私有参数)。"""
    std, extra = {}, {}
    for k, v in (params or {}).items():
        (std if k in _STANDARD_PARAMS else extra)[k] = v
    return std, extra


# --------------------------------------------------------------------------
# 错误分类：决定「要不要重试」
# --------------------------------------------------------------------------

_RETRYABLE_TRANSPORT = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def classify(exc: Exception) -> tuple[ErrorKind, bool]:
    """返回 (错误类别, 是否可重试)。"""
    # TypeError / ValueError 是本地参数或代码问题，重试无用 —— 必须先修配置
    if isinstance(exc, (TypeError, ValueError, AttributeError)):
        return ErrorKind.CLIENT, False
    if isinstance(exc, openai.RateLimitError):
        return ErrorKind.QUOTA, True
    if isinstance(exc, _RETRYABLE_TRANSPORT):
        return ErrorKind.TRANSPORT, True
    if isinstance(exc, openai.APIStatusError):
        # 5xx 可重试；其余 4xx 不重试（重试也不会变好，只会烧额度）
        if 500 <= exc.status_code < 600:
            return ErrorKind.TRANSPORT, True
        return ErrorKind.CLIENT, False
    if isinstance(exc, openai.BadRequestError):
        return ErrorKind.CLIENT, False
    return ErrorKind.TRANSPORT, True


def backoff_s(attempt: int, kind: ErrorKind) -> float:
    """指数退避。quota 退得更久，因为限流通常有窗口。"""
    base = 1.0 if kind is ErrorKind.TRANSPORT else 2.0
    return base * (2 ** attempt)


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------

class ChatClient:
    """薄封装。每次调用返回一个 ModelResult（永不抛异常——错误也落盘）。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.max_retries = max_retries
        resolved_base = base_url or paths.base_url()
        self._http = build_http_client(timeout_s, base_url=resolved_base)
        self._client = openai.OpenAI(
            base_url=resolved_base,
            api_key=api_key or paths.api_key(),
            timeout=timeout_s,
            max_retries=0,           # ★ 关键：自己管重试
            http_client=self._http,  # ★ 注入会话头 + 本地端点绕代理
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, spec: PromptSpec) -> ModelResult:
        """发一次请求。成功与失败都返回 ModelResult（失败时 success=False）。"""
        res = ModelResult(
            instance_id=spec.instance_id,
            arm=spec.arm,
            config=spec.config,
            rep=spec.rep,
            params=dict(spec.params),
            # ★ 必须把 target_arm 带进结果：judge 场景下三个模型的
            #   (config, arm, instance_id, rep) 完全相同，只有它能区分"在判哪个模型"。
            #   漏掉它会让三条 judge 记录塌成一条、判分时全归到 "unknown"。
            target_arm=spec.target_arm,
        )
        res.request_datetime = datetime.now(timezone.utc).isoformat()

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                std, extra = split_params(spec.params)
                kwargs: dict = {
                    "model": spec.arm,
                    "messages": [{"role": "user", "content": spec.prompt_text}],
                    **std,
                }
                if extra:
                    # 厂商私有参数（thinking / reasoning_effort…）必须走 extra_body，
                    # 直接当 kwarg 传会在本地抛 TypeError。
                    kwargs["extra_body"] = extra

                resp = self._client.chat.completions.create(**kwargs)
                res.latency_s = round(time.perf_counter() - t0, 3)
                self._fill(res, resp)
                return res

            except Exception as exc:  # noqa: BLE001 —— 错误也要变成数据
                last_exc = exc
                kind, retryable = classify(exc)
                res.error_kind = kind
                res.error = f"{type(exc).__name__}: {exc}"[:500]
                res.http_status = getattr(exc, "status_code", None)
                res.latency_s = round(time.perf_counter() - t0, 3)

                if not retryable or attempt >= self.max_retries:
                    res.success = False
                    return res
                time.sleep(backoff_s(attempt, kind))

        res.success = False
        res.error = f"exhausted retries: {last_exc}"[:500]
        return res

    @staticmethod
    def _fill(res: ModelResult, resp: object) -> None:
        """把 SDK 返回摊平进 ModelResult。字段缺失一律给安全默认值。"""
        choice = resp.choices[0]
        msg = choice.message

        res.success = True
        res.http_status = 200
        # ★ 必须清掉上一次尝试留下的错误痕迹。
        #   重试循环复用的是**同一个** ModelResult 对象，异常分支会写 error / error_kind；
        #   成功时若只设 success=True，就会出现「success=True + http_status=200 +
        #   content 非空 + 却带着 transport 错误标签」的记录 —— 实测历史上有 60/5,769 条。
        #   success 判定本身没错（所以结论不受影响），但任何按 error 非空统计的分析
        #   都会得到假阳性，且与门禁的 by_error_kind 聚合口径冲突。
        res.error = ""
        res.error_kind = None
        res.content = (getattr(msg, "content", None) or "")
        res.resp_model = getattr(resp, "model", "") or ""
        res.finish_reason = getattr(choice, "finish_reason", "") or ""

        # 思考内容：不同厂商字段名不一（reasoning_content / reasoning）
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(msg, "reasoning", None)
        res.reasoning = reasoning or ""

        usage = getattr(resp, "usage", None)
        if usage is not None:
            res.tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            res.tokens_out = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                res.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
