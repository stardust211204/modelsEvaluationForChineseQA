"""② 请求构造层：Instance × Config × Arm → PromptSpec。

铁律：**纯函数**。同样的输入必得 byte-identical 的 prompt。
这是断点续跑能成立的前提，也是 C1/C2/C5 配对比较能成立的前提。

本层不做的事：
  * 不发网络请求
  * 不读模型输出
  * 不做任何随机（few-shot 示例必须是预先固定好的 id 列表，不是运行时抽的）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import build_set, paths
from .types import Config, Instance, PromptSpec, Split

# --------------------------------------------------------------------------
# 配置对象
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    name: str
    vendor: str
    model: str
    concurrency: int = 4
    params: dict[str, Any] = field(default_factory=dict)
    # ★ 模型级端点覆盖。
    #   默认 None ⇒ 走 `paths.base_url()`（聚合端点）。
    #   为什么要它：旗舰档 `deepseek-v4-pro` 走**官方端点**（聚合端点该模型额度少），
    #   于是同一批评测里**不同模型落在不同端点**。若客户端仍是全局单例，
    #   这个差异就表达不出来 —— 只能拆成两个 run，那样模型之间又不可配对了。
    #
    #   ⚠️ 这会让"模型差异"与"端点差异"混在一起。已有的缓解证据：
    #      `runs/endpoint-cmp` 实测 官方 deepseek-flash ≈ 聚合 deepseek-v4.1-flash
    #      （96/100 token 完全一致）。但**那只验证了 flash 档**，
    #      v4-pro 的端点一致性未验证 —— 报告里必须声明。
    base_url: str | None = None
    key_env: str | None = None      # 该模型用哪个环境变量取 key；None ⇒ OPENCODE_API_KEY
    # ★ 客户端参数：原先 `load_arms` 把这两个键从 params 里
    #   `pop` 出来**直接丢弃** ⇒ `configs/arms.yaml` 的 `timeout_s: 120` 一直是摆设，
    #   实际生效的只是 `client.py` 的同值默认。后果是**改了这个配置项没有任何效果**，
    #   而开思考的请求实测会超过 120s（E1 实测 8/1000 条被掐断记成 transport 失败）。
    #   现在它们随 Arm 传递，由 run_config / run_specs 真正用上。
    #   None ⇒ 用 ChatClient 的默认值（120s / 4 次重试），与改动前逐字节相同。
    timeout_s: float | None = None
    max_retries: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name, "vendor": self.vendor, "model": self.model,
            "concurrency": self.concurrency, "params": self.params,
        }
        if self.base_url:
            d["base_url"] = self.base_url
        if self.key_env:
            d["key_env"] = self.key_env
        # 只在**显式设置**时出现 ⇒ 未设置的模型与历史记录逐字节一致
        if self.timeout_s is not None:
            d["timeout_s"] = self.timeout_s
        if self.max_retries is not None:
            d["max_retries"] = self.max_retries
        return d


@dataclass(frozen=True)
class ConfigSpec:
    key: Config
    name: str
    factor: str | None
    desc: str
    prompt_template: str
    n_instances: int
    sampling: str = "c1_subset"
    reps: int = 1
    params_override: dict[str, Any] = field(default_factory=dict)
    few_shot: dict[str, Any] | None = None
    judge: bool = False
    judge_arm: str | None = None
    # C6：按 rep 重排选项（rep 即「第几个排列」，0 = 原顺序）。
    option_permutation: bool = False
    rq: str = ""


# --------------------------------------------------------------------------
# 载入配置
# --------------------------------------------------------------------------

def load_arms(path: Path | None = None) -> tuple[dict[str, Any], list[Arm]]:
    """返回 (全局配置, 模型列表)。模型的 params 已与 defaults 合并。"""
    cfg = yaml.safe_load((path or paths.configs() / "arms.yaml").read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {}) or {}
    arms: list[Arm] = []
    for a in cfg.get("arms", []):
        params = {**defaults, **(a.get("params") or {})}
        # ★ 这两个是**客户端参数**，不属于模型参数 ⇒ 从 params 里取出。
        #   但**必须留着传给 Arm**：原先 pop 完就丢，等于让 arms.yaml 的
        #   `timeout_s` / `max_retries` 变成摆设（改了没有任何效果）。
        _to = params.pop("timeout_s", None)
        _mr = params.pop("max_retries", None)
        arms.append(
            Arm(
                name=a["name"], vendor=a.get("vendor", ""), model=a["model"],
                concurrency=int(a.get("concurrency", 4)), params=params,
                base_url=a.get("base_url"), key_env=a.get("key_env"),
                timeout_s=float(_to) if _to is not None else None,
                max_retries=int(_mr) if _mr is not None else None,
            )
        )
    return cfg, arms


def load_matrix(path: Path | None = None) -> list[ConfigSpec]:
    raw = yaml.safe_load((path or paths.configs() / "matrix.yaml").read_text(encoding="utf-8"))
    out: list[ConfigSpec] = []
    for key, c in (raw.get("configs") or {}).items():
        out.append(
            ConfigSpec(
                key=Config(key),
                name=c.get("name", ""),
                factor=c.get("factor"),
                desc=c.get("desc", ""),
                prompt_template=c["prompt_template"],
                n_instances=int(c["n_instances"]),
                sampling=c.get("sampling", "c1_subset"),
                reps=int(c.get("reps", 1)),
                params_override=c.get("params_override") or {},
                few_shot=c.get("few_shot"),
                judge=bool(c.get("judge", False)),
                judge_arm=c.get("judge_arm"),
                option_permutation=bool(c.get("option_permutation", False)),
                rq=c.get("rq", ""),
            )
        )
    return out


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------

def render_options(inst: Instance) -> str:
    """把选项渲染成 (A) xxx 形式。"""
    return "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(inst.options))


def render_prompt(inst: Instance, template: str, demos: list[Instance] | None = None) -> str:
    """把一个 Instance 填进模板。纯函数：无随机、无时间、无外部状态。

    ★ `{last}` = 该题**实际**的最后一个选项字母（4 选项 → D，9 选项 → I）。

      修的是一个会造成**模型间不公平**的提示词缺陷：三个模板原先都写死
      「必须且只能是一个大写字母（**A 到 J** 之一）」，而 MMLU-ProX 有
      103/588 题选项数 <10、C-Eval 更是 583/583 题只有 4 个选项。
      于是提示词告诉模型"可以答 J"，而选项里根本没有 J。

      实测（同题同模型、只改这一处，各 3 次）：

          提示词「A 到 J」 -> J, J, J     （不存在的选项，判 invalid）
          提示词「A 到 I」 -> I, I, I     （★ 而且 I 正是标准答案）

      ⇒ 受影响的是**严格照字面执行指令的模型**：旗舰 2.4% 越界，
        而四个 flash 模型 0–1.5%（它们会自己看选项列表）。
        这不是"模型差异"，是**提示词把一部分模型带进了坑**。

    ★ `{demos}` 必须在 `.format()` **之后**才填进去。

      原先的写法是先 `text.replace("{demos}", render_demos(...))` 再对**整个字符串**
      调 `.format()`。后果：**示例内容里的花括号被当成占位符**。

      实测（E5 的 n=10 一档）：示例 `prox-zh-valid-0` 是一道组合数学题，题面含
      `{n}` 与 `{10}` ⇒ `text.format()` 抛 `KeyError: 'n'`，**整个 run 崩掉**。
      valid 池 70 条里有 2 条含花括号（`valid-0`、`valid-2`）。

      为什么主跑批从未暴露：题目内容（`question`/`options`）是作为 `.format()` 的
      **参数**传入的，`str.format` 只解析**格式串**、不解析参数 ⇒ 题目里的花括号无害。
      **只有 `{demos}` 这一条路径会把数据拼进格式串。** 而 `{demos}` 只出现在
      `prompts/c5_fewshot.txt` ⇒ 只有 few-shot 配置会踩到。

      ⇒ 用哨兵把示例的插入挪到 `.format()` 之后：**数据永远不参与模板解析**。
    """
    # 哨兵用 \x00 包裹：正常文本里不可能出现，故不会与模板内容冲突。
    demos_slot = "\x00DEMOS\x00"
    text = template.replace("{demos}", demos_slot)
    out = text.format(
        question=inst.question,
        options=render_options(inst),
        last=chr(64 + len(inst.options)),   # ★ 按实际选项数，不再写死 J
    )
    if demos_slot in out:
        out = out.replace(demos_slot, render_demos(demos or []))
    return out


def render_judge_prompt(inst: Instance, response: str, template: str) -> str:
    """渲染 judge 提示词（C4）。

    ★ 用 `str.replace` 而不是 `str.format`：
      `{response}` 要塞进**模型自己的输出**，而那段文本里完全可能出现花括号
      （公式、JSON 片段、LaTeX）。用 `format` 会当场抛 KeyError/ValueError，
      而且报错指向模板，不指向真正的原因 —— 极难定位。

    ★ judge 模板需要 `{gold}` 与 `{response}` 两个 `render_prompt` 不提供的占位符。
      这正是 C4 长期跑不起来的直接原因（`KeyError: 'gold'`），
      也是"直接把 judge 模板丢给被测模型"这个错误实验的入口。
    """
    return (template
            .replace("{question}", inst.question)
            .replace("{options}", render_options(inst))
            .replace("{gold}", "/".join(inst.gold_letters))
            .replace("{response}", response))


def render_demos(demos: list[Instance]) -> str:
    """渲染 few-shot 示例块。

    示例必须来自 validation split（见 docs/design.md）—— 取自 test 即数据泄漏。

    ★ 答案行必须是**裸字母**，不能写成「答案：B」。
      示例对输出格式的示范作用**强于正文里的文字要求**：写成「答案：B」
      会诱导模型输出带前缀、带解释的长回复，于是"整段回复只能是单个字母"
      的严格口径直接失效 —— C5 首轮就是这样整轮作废的（393/450 判不出来）。
      因此每个示例的**最后一行**必须正好是那道题应当输出的全部内容。
    """
    blocks: list[str] = []
    for i, d in enumerate(demos, 1):
        gold = "".join(chr(65 + g) for g in d.gold_indices)
        blocks.append(
            f"示例 {i}\n题目：{d.question}\n{render_options(d)}\n{gold}\n"
        )
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------

def build_specs(
    *,
    run_id: str,
    config: ConfigSpec,
    arms: list[Arm],
    instances: list[Instance],
    demos: list[Instance] | None = None,
    prompt_template: str | None = None,
    arm_filter: list[str] | None = None,
) -> list[PromptSpec]:
    """展开成待执行的 PromptSpec 列表。

    ★ 顺序按「模型交错」（instance 在外层、arm 在内层），不是「模型在外层」。
      原因：`--limit N` 是取列表前 N 条做缩量试跑。若按模型在外层排序，
      前 N 条会全部落在第一个模型上 —— 试跑就验证不到跨模型并发，
      各模型的错误率与延迟也拿不到。交错后前 N 条自然覆盖所有模型。

    顺序仍是确定性的（同一输入必得同一顺序），因此 diff 与复现不受影响。
    """
    tmpl = prompt_template
    if tmpl is None:
        tmpl = (paths.ROOT / config.prompt_template).read_text(encoding="utf-8")

    selected = [a for a in arms if not arm_filter or a.name in arm_filter]
    specs: list[PromptSpec] = []

    for inst in instances:
        for rep in range(config.reps):
            # C6：rep 就是「第几个排列」，k=0 为原顺序（等价于 C1 的题面）。
            # 排列是 (instance_id, k) 的确定性函数，所以断点续跑仍然安全。
            use_inst = (build_set.permute_instance(inst, rep)[0]
                        if config.option_permutation else inst)
            prompt = render_prompt(use_inst, tmpl, demos)
            for arm in selected:
                params = dict(arm.params)
                params.update(config.params_override)
                specs.append(
                    PromptSpec(
                        run_id=run_id,
                        config=config.key,
                        arm=arm.model,
                        instance_id=inst.id,
                        rep=rep,
                        prompt_text=prompt,
                        demos=[d.id for d in (demos or [])],
                        params=params,
                    )
                )
    return specs


def prompt_fingerprint(
    instances: list[Instance],
    template: str,
    demos: list[Instance] | None = None,
    permutations: int = 1,
) -> str:
    """把一批题渲染后的 prompt 串起来取哈希。

    两个用途，都是"防静默失效"：

    ① 门禁：题集哈希没变、模型参数没变，但**模板或渲染逻辑变了** ——
       跨 run 的配对比较同样失效，而单看任何一份产物都发现不了。
    ② 跑批：指纹写进 manifest，resume 时对不上就拒跑。原因见下。

    ★ 断点续跑的键是 `(config, arm, instance_id, rep)`，**不含 prompt**。
      改了模板却沿用同一 run_id，`scan_done()` 会把全部请求当成"已完成"跳过，
      于是产出一批用旧模板跑出来的数据，而且**不报任何错**
      —— 与"把 0-shot 跑成 5-shot"同属一类静默降级。

    `permutations > 1` 时把每个排列都算进去（C6）。**`permutations == 1` 的分支
    刻意保持与旧实现逐字节相同** —— 否则所有已冻结的指纹都会失效、
    门禁会把"没改任何东西"误报成漂移。
    """
    h = hashlib.sha256()
    for inst in instances:
        if permutations > 1:
            for k in range(permutations):
                use = build_set.permute_instance(inst, k)[0]
                h.update(f"{inst.id}#{k}".encode("utf-8"))
                h.update(b"\x00")
                h.update(render_prompt(use, template, demos).encode("utf-8"))
                h.update(b"\x01")
        else:
            h.update(inst.id.encode("utf-8"))
            h.update(b"\x00")
            h.update(render_prompt(inst, template, demos).encode("utf-8"))
            h.update(b"\x01")
    return h.hexdigest()[:16]


def assert_pure(fn, *args, **kwargs) -> None:
    """开发期自检：同一输入两次调用必须产出完全相同结果。

    在 CI 或 F2 验证时调用一次即可 —— 它拦住的是"运行时悄悄抽了 few-shot"
    这类最难查的不可复现问题。
    """
    a = fn(*args, **kwargs)
    b = fn(*args, **kwargs)
    sa = json.dumps([s.to_dict() for s in a], ensure_ascii=False, sort_keys=True)
    sb = json.dumps([s.to_dict() for s in b], ensure_ascii=False, sort_keys=True)
    if sa != sb:
        raise AssertionError("build_specs 不是纯函数：两次调用结果不同")


# --------------------------------------------------------------------------
# 子集抽样（C2 / C3 / C5 必须取自 C1）
# --------------------------------------------------------------------------

def subset_from_c1(c1_instances: list[Instance], n: int) -> list[Instance]:
    """从 C1 的 600 题里按固定间隔取 n 题，保证：
      ① 是 C1 的真子集（配对比较的前提）
      ② 跨难度档均匀（间隔取样天然跨 strata）
      ③ 确定性可复现（不用随机数）
    """
    total = len(c1_instances)
    if n >= total:
        return list(c1_instances)
    step = total / n
    idxs = sorted({int(i * step) for i in range(n)})
    # 补齐（浮点取整可能少一两个）
    cursor = 0
    while len(idxs) < n and cursor < total:
        if cursor not in idxs:
            idxs.append(cursor)
        cursor += 1
    return [c1_instances[i] for i in sorted(idxs)[:n]]


def load_demos(
    c5: ConfigSpec,
    valid_pool: list[Instance],
    dataset: str | None = None,
) -> list[Instance]:
    """按 C5 配置取固定的 few-shot 示例。

    只在 split == valid 的池子里取。取不到就报错 —— 不允许静默从 test 取。

    ★ `fixed_ids_by_dataset`（2026-10-06 加）
      同一配置在不同评测集上用不同的示例组。用途：让示例的**选项数**与目标题匹配。

      为什么需要它：MMLU-ProX 以 10 选项为主（485/588），而 C-Eval **全部 583 题
      都是 4 选项**。原先两个评测集共用同一组示例（选项数 10/10/10/8/4，平均 8.4），
      在 C-Eval 上就成了一次「10 选项示例 → 4 选项题」的跨格式迁移实验 ——
      C5/C-Eval 上 ds-pro 掉 13.3 个百分点，无法区分是「示例数量」还是
      「示例格式错配」造成的。

      `dataset` 为 None 时（如门禁做整体指纹时）退回 `fixed_ids`。
    """
    fs = c5.few_shot or {}
    n = int(fs.get("n", 5))
    fixed = fs.get("fixed_ids") or []
    by_ds = fs.get("fixed_ids_by_dataset") or {}

    if dataset and dataset in by_ds:
        fixed = by_ds[dataset]

    pool = [i for i in valid_pool if i.split is Split.VALID]
    if not pool:
        raise RuntimeError(
            "few-shot 示例池为空。示例必须取自 validation split —— "
            "从 test split 取示例即数据泄漏，本项目不允许。"
        )
    if fixed:
        by_id = {i.id: i for i in pool}
        missing = [i for i in fixed if i not in by_id]
        if missing:
            raise RuntimeError(f"fixed_ids 中有 {len(missing)} 条不在 valid 池中: {missing[:3]}")
        return [by_id[i] for i in fixed][:n]

    # 未指定 fixed_ids：按 id 排序取前 n（确定性，非随机）
    return sorted(pool, key=lambda i: i.id)[:n]
