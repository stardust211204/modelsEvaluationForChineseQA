"""统一路径解析。

唯一的路径真相来源。任何模块都不得再自己用 `Path(__file__).parent...` 推算目录
——那正是迁移时最容易写错位置的地方（迁移前的脚本会把产物写到项目根之外）。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# src/meval/paths.py -> src/meval -> src -> <root>
ROOT: Path = Path(__file__).resolve().parents[2]


def _dir(*parts: str) -> Path:
    p = ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- 静态目录 -------------------------------------------------------------

def configs() -> Path:
    return _dir("configs")


def docs() -> Path:
    return _dir("docs")


def data() -> Path:
    return _dir("data")


def evidence() -> Path:
    """迁移过来的探测原始记录（实验证据）。"""
    return _dir("data", "raw_evidence")


def figures() -> Path:
    return _dir("figures")


def reports() -> Path:
    """对外/对内报告与门禁产物的落盘目录。"""
    return _dir("reports")


def tests() -> Path:
    return _dir("tests")


# ---- 跑批目录 -------------------------------------------------------------

def new_run_id(prefix: str = "run") -> str:
    """UTC 时间戳式的 run_id，保证可排序且不冲突。"""
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def run_dir(run_id: str) -> Path:
    return _dir("runs", run_id)


def raw_dir(run_id: str, config: str) -> Path:
    """不可变的调用事实。"""
    return _dir("runs", run_id, "raw", config)


def tallies_dir(run_id: str, config: str) -> Path:
    """可重算的判分结论。"""
    return _dir("runs", run_id, "tallies", config)


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"


# ---- 环境 -----------------------------------------------------------------

# ★ .env.example 一直写着"复制为 .env 后填入真实值"，
#   但**此前没有任何代码读它** —— 文档描述的能力并不存在。
#   这里补上：只在环境变量缺失时填充，**绝不覆盖**已有的
#   （Windows 用户级环境变量会被继承进 os.environ，因此优先级自然更高，
#   也便于临时切换 key 而不动文件）。
_DOTENV_LOADED = False


def _load_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    p = ROOT / ".env"
    if not p.exists():
        return
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


def env(key: str, default: str | None = None) -> str | None:
    """读环境变量：进程环境（含 .env 填充）→ 用户级环境变量（Windows）。"""
    _load_dotenv()
    val = os.environ.get(key)
    if val:
        return val
    if os.name == "nt":
        try:
            import winreg  # noqa: PLC0415

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                return winreg.QueryValueEx(k, key)[0]
        except OSError:
            pass
    return default


def api_key() -> str:
    key = env("OPENCODE_API_KEY")
    if not key:
        raise RuntimeError(
            "未找到 OPENCODE_API_KEY。请设置环境变量，或复制 .env.example 为 .env 并填写。"
        )
    return key


def base_url() -> str:
    return env("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1") or ""


# ---- judge 端点（与三个模型的聚合端点**分开**）--------------------------------
# judge 必须与被评模型尽量不同源：见 docs/OPEN_ISSUES.md P2-1。

def judge_api_key() -> str:
    key = env("DEEPSEEK_API_KEY") or env("JUDGE_API_KEY")
    if not key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY / JUDGE_API_KEY —— judge 走独立端点，"
            "请写入 .env（已在 .gitignore 中）或设为环境变量。"
        )
    return key


def judge_base_url() -> str:
    return (env("DEEPSEEK_BASE_URL") or env("JUDGE_BASE_URL")
            or "https://api.deepseek.com") or ""


def judge_model() -> str:
    return env("JUDGE_MODEL", "deepseek-v4-pro") or "deepseek-v4-pro"
