"""③ 执行层：并发调度 + 退避重试 + 断点续跑 + 原始落盘。

铁律：**本层不解读模型输出。** 它只负责"把请求发出去、把原文存下来"。
任何判分逻辑出现在这里都是越界 —— 见 docs/design.md §3。

断点续跑的关键：请求是**无状态**的（每题独立、无对话历史），
所以跳过已完成的题不会改变任何结果。这是能安全续跑的根据。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from . import paths
from .client import ChatClient
from .types import Config, ErrorKind, ModelResult, PromptSpec, RunManifest

# --------------------------------------------------------------------------
# 已完成键集合（断点续跑的依据）
# --------------------------------------------------------------------------

Key = tuple[str, str, str, int, str]  # (config, arm, instance_id, rep, target_arm)


def _key_of(d: dict) -> Key:
    """去重键。

    ★ 含 `target_arm`：C4 要对三个模型的同一批题各判一次，而三次调用的
      (config, arm, instance_id, rep) 完全相同（arm 都是 judge 模型）。
      键里不带 target_arm 就会三个模型互相覆盖、只剩最后一个模型 ——
      结果字典的键漏掉 arm 就会互相覆盖 —— 这个坑在本项目里踩过两次。
    """
    return (str(d.get("config")), str(d.get("arm")), str(d.get("instance_id")),
            int(d.get("rep", 0)), str(d.get("target_arm", "")))


def scan_done(run_id: str) -> set[Key]:
    """扫描已落盘的 raw JSONL，返回「已完成」键集合。

    规则：同键出现多次时以**最后一次**为准；只有最后一次成功才算已完成
    （so 之前失败、之后重试成功的条目不会被重复调用）。
    尾部的半截行（崩溃残留）会被忽略。
    """
    latest: dict[Key, bool] = {}
    root = paths.ROOT / "runs" / run_id / "raw"
    if not root.exists():
        return set()
    for f in root.rglob("*.jsonl"):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # 半截行，忽略
            latest[_key_of(d)] = bool(d.get("success"))
    return {k for k, ok in latest.items() if ok}


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------

class RawWriter:
    """按 (config, arm) 分文件追加写。单写锁 + 每条 flush，崩溃不丢已完成数据。"""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._lock = threading.Lock()
        self._handles: dict[Path, object] = {}

    def _handle(self, config: Config, arm: str):
        path = paths.raw_dir(self.run_id, config.value) / f"{_safe(arm)}.jsonl"
        h = self._handles.get(path)
        if h is None:
            h = path.open("a", encoding="utf-8")
            self._handles[path] = h
        return h

    def write(self, res: ModelResult) -> None:
        line = json.dumps(res.to_dict(), ensure_ascii=False)
        with self._lock:
            h = self._handle(res.config, res.arm)
            h.write(line + "\n")
            h.flush()

    def close(self) -> None:
        with self._lock:
            for h in self._handles.values():
                try:
                    h.close()
                except OSError:
                    pass
            self._handles.clear()


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------

def run_specs(
    *,
    run_id: str,
    specs: list[PromptSpec],
    arms_concurrency: dict[str, int],
    client: ChatClient | None = None,
    arm_endpoints: dict[str, tuple[str | None, str | None]] | None = None,
    arm_timeouts: dict[str, float] | None = None,
    arm_retries: dict[str, int] | None = None,
    limit: int | None = None,
    progress: bool = True,
    timeout_s: float | None = None,
) -> dict[str, object]:
    """执行一批 PromptSpec，返回统计摘要。

    arms_concurrency: {arm_model: 并发度}。不同模型可以不同（免费档保守、付费档放开）。

    arm_endpoints: {arm_model: (base_url, key_env)}。**仅当某模型需要走与全局不同的端点时**才传。
        ★ 为什么按模型而非按 run：旗舰档走官方端点、其余模型走聚合端点时，
          若拆成两个 run，"模型"和"接入方式"就同时变了，违反单因子纪律；
          合成一个 run + 模型级端点，至少让这个差异**显式记录在 manifest 里**。

    timeout_s: 客户端超时（秒）。**默认 None ⇒ 用 ChatClient 的默认值（120s），
        与改动前逐字节相同。**
        ★ 为什么需要它：补充验证实验里 `reasoning_effort=high` 的请求实测会超过
          120s（推理最长的那几条），被掐断后记成 `transport` 失败。
          而 `configs/arms.yaml` 的 `timeout_s: 120` 被 `adapt.load_arms()`
          弹出后**从未传给客户端** —— 那个配置项一直是摆设，实际生效的
          只是 `client.py` 的同值默认。
    """
    own_client = client is None
    # ★ 模型级客户端池。没有 arm_endpoints / arm_timeouts / arm_retries 时
    #   退化成"所有模型共用一个客户端"（即原行为），所以这是纯增量改动。
    _ckw = {"timeout_s": timeout_s} if timeout_s is not None else {}
    _pool: dict[str, ChatClient] = {}
    _shared = client if client is not None else ChatClient(**_ckw)

    def client_for(arm: str) -> ChatClient:
        ep = (arm_endpoints or {}).get(arm)
        # 超时优先级：run 级 `timeout_s` 覆盖 > 模型级 `arm_timeouts` > ChatClient 默认。
        # ★ 模型级这一路原先**根本不存在** —— `adapt.load_arms` 把 arms.yaml 的
        #   `timeout_s` 弹出后丢弃，所以那个配置项改了没有任何效果。
        to = timeout_s if timeout_s is not None else (arm_timeouts or {}).get(arm)
        mr = (arm_retries or {}).get(arm)
        # 三者都没有 ⇒ 沿用共享客户端（原行为，逐字节相同）
        if client is not None or (not ep and to is None and mr is None):
            return _shared
        if arm not in _pool:
            burl, kenv = ep or (None, None)
            key = None
            if kenv:
                key = paths.env(kenv)
                if not key:
                    raise RuntimeError(f"模型 {arm} 需要环境变量 {kenv}，但未设置")
            kw = dict(_ckw)
            if to is not None:
                kw["timeout_s"] = to
            if mr is not None:
                kw["max_retries"] = mr
            _pool[arm] = ChatClient(base_url=burl, api_key=key, **kw)
        return _pool[arm]

    def close_all() -> None:
        for c in _pool.values():
            c.close()
        if own_client:
            _shared.close()

    writer = RawWriter(run_id)

    done = scan_done(run_id)
    todo = [s for s in specs if s.key not in done]
    skipped = len(specs) - len(todo)
    # ★ `if limit:` 会把 `--limit 0` 当成"没限制" ⇒ 意外跑全量。
    #   0 应当是"一条都不跑"（校验/干跑场景会用），所以显式判 None。
    if limit is not None:
        todo = todo[:limit]

    started = datetime.now(timezone.utc).isoformat()
    stats: dict[str, object] = {
        "run_id": run_id,
        "total_specs": len(specs),
        "skipped_done": skipped,
        "executed": 0,
        "failed": 0,
        "by_error_kind": {},
        "by_arm": {},
    }

    if not todo:
        print(f"[execute] 无待执行请求（已完成 {skipped} 条）")
        writer.close()
        close_all()
        return stats

    print(f"[execute] 总 {len(specs)} 条｜跳过已完成 {skipped} 条｜本次执行 {len(todo)} 条")

    # 按模型各自限流：把任务按模型分组，每个模型用一个独立线程池，最后统一等
    by_arm: dict[str, list[PromptSpec]] = {}
    for s in todo:
        by_arm.setdefault(s.arm, []).append(s)

    lock = threading.Lock()
    # 非 TTY（重定向到文件 / 被 pwsh 包装）时不画进度条：
    # tqdm 的 \r 刷新会被 PowerShell 当成 stderr 异常，污染日志并让退出码看起来是 1。
    use_bar = progress and sys.stderr.isatty()
    pbar = tqdm(total=len(todo), disable=not use_bar, desc="执行", unit="req")

    def one(spec: PromptSpec) -> ModelResult:
        return client_for(spec.arm).call(spec)

    def consume(res: ModelResult) -> None:
        writer.write(res)
        with lock:
            stats["executed"] = int(stats["executed"]) + 1
            if not res.success:
                stats["failed"] = int(stats["failed"]) + 1
                kinds = stats["by_error_kind"]
                k = res.error_kind.value if res.error_kind else "unknown"
                kinds[k] = kinds.get(k, 0) + 1
            per = stats["by_arm"]
            a = per.setdefault(res.arm, {"n": 0, "failed": 0, "secs": 0.0, "tokens_out": 0})
            a["n"] += 1
            a["failed"] += int(not res.success)
            a["secs"] += res.latency_s
            a["tokens_out"] += res.tokens_out
            pbar.update(1)

    try:
        futures = []
        with ThreadPoolExecutor(max_workers=sum(arms_concurrency.get(a, 4) for a in by_arm)) as pool:
            # 用信号量把每个模型的并发压到各自上限
            sems = {a: threading.Semaphore(arms_concurrency.get(a, 4)) for a in by_arm}

            def guarded(spec: PromptSpec) -> ModelResult:
                sem = sems[spec.arm]
                with sem:
                    return one(spec)

            for spec in todo:
                futures.append(pool.submit(guarded, spec))
            for fut in as_completed(futures):
                consume(fut.result())
    finally:
        pbar.close()
        writer.close()
        close_all()

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    stats["started_at"] = started
    print(
        f"[execute] 完成：成功 {int(stats['executed']) - int(stats['failed'])}｜"
        f"失败 {stats['failed']}｜错误分布 {stats['by_error_kind']}"
    )
    return stats


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def write_manifest(
    *,
    run_id: str,
    configs: list[str],
    arms: list[dict],
    data_hash: str,
    models_snapshot: list[str] | None = None,
    note: str = "",
    stats: dict | None = None,
    prompt_fingerprints: dict[str, str] | None = None,
    param_overrides: dict | None = None,
    reps: int | None = None,
    param_arms: list[str] | None = None,
    dataset: str = "eval_set",
) -> Path:
    """写跑批元数据。让每个数字可回溯到哪次跑批、哪版代码、哪版数据。

    ★ 断点续跑时要**并入**已记录的模型，不能覆盖。

      场景：`c1-4arm` 已跑完四个模型，现在补第 5 个模型（旗舰档）。若直接写，
      `manifest.arms` 就只剩 `deepseek-v4-pro` —— 而 raw/tallies 里明明有五个模型。
      后果不是"报错"，而是**静默失真**：门禁的模型别名检查、后人的回溯
      都会以为这个 run 只有一个模型。数据在、记录不在，是最难查的一类不一致。

    `dataset` 声明本次跑批用的是**哪份题集**。默认 `eval_set`（即
    `data/eval_set.jsonl`，绝大多数 run 都是它）。

    ★ 为什么需要这个字段：门禁会把 manifest 的 `data_hash` 与**冻结的 eval_set 哈希**
      比对，不一致就判 FAIL。而 C7 拒答评测用的是另一份题集
      （`data/refusal_set.jsonl`）—— 若不在 manifest 里声明，
      门禁会把"用了另一份数据集"误报成"题集被改了，此 run 不可配对"。
      比"把 eval_set 的哈希填进去骗过门禁"诚实，也比"干脆不写 manifest"可追溯。
    """
    p = paths.manifest_path(run_id)
    new_arms = [a["name"] for a in arms]
    merged_arms, merged_params, merged_models = list(new_arms), \
        {a["name"]: a.get("params", {}) for a in arms}, list(models_snapshot or [])
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
        for name in old.get("arms") or []:
            if name not in merged_arms:
                merged_arms.append(name)
        op = old.get("params_effective") or {}
        merged_params = {**op, **merged_params}   # 新值覆盖旧值（同模型重跑时参数可能变了）
        for mdl in old.get("models_snapshot") or []:
            if mdl not in merged_models:
                merged_models.append(mdl)

    m = RunManifest(
        run_id=run_id,
        configs=configs,
        arms=merged_arms,
        params_effective=merged_params,
        data_hash=data_hash,
        prompt_fingerprints=prompt_fingerprints or {},
        param_overrides=param_overrides or {},
        reps=reps,
        param_arms=param_arms,
        code_version=_code_version(),
        models_snapshot=merged_models,
        started_at=(stats or {}).get("started_at", datetime.now(timezone.utc).isoformat()),
        finished_at=(stats or {}).get("finished_at", ""),
        note=note,
        dataset=dataset,
    )
    p.write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _code_version() -> str:
    """尽力取 git 短哈希；没有仓库就返回 'nogit'。"""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


# --------------------------------------------------------------------------
# 读取 raw（供 tally / metrics 使用）
# --------------------------------------------------------------------------

def load_results(run_id: str, config: Config | None = None) -> list[ModelResult]:
    """读回 raw。同键多条时取最后一条成功记录（失败记录只在无成功记录时保留）。"""
    root = paths.ROOT / "runs" / run_id / "raw"
    latest: dict[Key, ModelResult] = {}
    if not root.exists():
        return []
    for f in sorted(root.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if config and d.get("config") != config.value:
                continue
            d["config"] = Config(d["config"])
            if d.get("error_kind"):
                d["error_kind"] = ErrorKind(d["error_kind"])
            r = ModelResult(**d)
            k: Key = (r.config.value, r.arm, r.instance_id, r.rep, r.target_arm)
            prev = latest.get(k)
            if prev is None or (r.success and not prev.success):
                latest[k] = r
    return list(latest.values())
