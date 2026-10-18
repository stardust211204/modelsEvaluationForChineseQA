"""回归门禁：把「这个仓库当前是否自洽」变成一条可执行的命令。

为什么需要它
    本项目报告里的每个数字都依赖三条**隐含前提**：
      ① 题集冻结后没被改过 —— 改了则 C1/C2/C3/C5 之间不可配对比较
      ② 三个模型参数是冻结的那一套 —— 漂了则跨 run 比较无意义
      ③ 严格/宽松两口径在**二元判分**上不产生分歧 ——
         一旦分歧，"判分口径不制造伪低分"这个核心结论就不成立
    这三条被破坏时，报告里的数字**看起来仍然正常**。静默失效是最难查的，
    门禁把它们变成硬断言，让"坏了"表现为一条 FAIL 而不是一个漂亮的错数。

用法
    .venv\\Scripts\\python.exe scripts\\ci_gate.py                  # 全部门禁
    .venv\\Scripts\\python.exe scripts\\ci_gate.py --skip-tests      # 跳过 pytest
    .venv\\Scripts\\python.exe scripts\\ci_gate.py --freeze          # 重新记录冻结指纹
    .venv\\Scripts\\python.exe scripts\\ci_gate.py --json            # 另写机器可读报告

退出码：0 = 无 FAIL；1 = 至少一条 FAIL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, build_set, execute, metrics, paths  # noqa: E402
from meval.types import Config, Split, Source  # noqa: E402

FROZEN = "frozen.json"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------

def sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def file_sha16(p: Path) -> str:
    return sha16(p.read_bytes())


def norm(obj) -> str:
    """归一化成可比较的字符串（键排序，忽略 dict 内顺序差异）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def is_smoke(run_id: str) -> bool:
    """演练类 run：题集与全量不同是**设计如此**，不该按全量标准判 FAIL。"""
    return "smoke" in run_id or "pilot" in run_id


def raw_available() -> bool:
    """仓库里是否存在**任何** raw 数据。

    raw 不进 git（75 MB，走 GitHub Release 附件，见 README §六），
    所以**刚 clone 下来的仓库一条 raw 都没有**。此时「raw 完整性」「judge tally
    完整性」这类检查的**输入根本不存在**，应当报 SKIP 而不是 FAIL ——
    否则面试官照 README 跑一遍会看到「门禁结果：不通过 —— 有结论级前提被破坏」，
    而真相只是没下载数据包。

    ★ 判据必须是**全局**的：只要有一个 `runs/*/raw` 存在，就说明数据已下载，
      此时某个批次缺 raw 是**真失败**，必须照报 FAIL。不能按单批次判断，
      否则「跑批跑丢了」这种真事故会被静默降级成 SKIP。
    """
    return any((paths.ROOT / "runs").glob("*/raw"))


_RAW_AVAILABLE = raw_available()
_RAW_HINT = "raw 未下载（走 Release 附件，见 README §六）"


class Gate:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append({"check": name, "status": status, "detail": detail})

    def ok(self, name: str, detail: str = "") -> None:
        self.add(name, PASS, detail)

    def bad(self, name: str, detail: str = "") -> None:
        self.add(name, FAIL, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.add(name, WARN, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.add(name, SKIP, detail)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r["status"] == FAIL)


# --------------------------------------------------------------------------
# 期望值推算（与 run_config.py 保持同一套规则）
# --------------------------------------------------------------------------

def expected_specs(pool: list, spec: adapt.ConfigSpec, n_arms: int,
                   reps: int | None = None) -> int:
    """一个 run 一个 config 应有多少条请求 —— 用与跑批**同一套**抽样规则算。

    ★ `reps`：manifest 显式声明了实际重复次数时以它为准。
      原先只读 matrix.yaml 的 `spec.reps`，于是补充验证实验用 `--reps 1`
      （如 E1 思考开关，矩阵里没有「100 题 + reps=1」的组合）时，
      门禁会按 c3 的 reps=10 判成「缺 4500 条」—— 把有意的变体误报成数据缺失。
      历史 manifest 无该键 ⇒ 回退到 matrix 值，行为不变。
    """
    if spec.key is Config.C1:
        insts = list(pool)
    else:
        insts = adapt.subset_from_c1(pool, spec.n_instances)
    return len(insts) * n_arms * (reps if reps is not None else spec.reps)


def prompt_fingerprint(pool: list, spec: adapt.ConfigSpec, demos=None) -> str:
    """一个配置的 prompt 指纹（题集切片 + 模板 + 示例 [+ 排列]）。

    ★ 哈希逻辑在 `adapt.prompt_fingerprint` —— 门禁与跑批必须用**同一份**实现，
      否则会出现"门禁说没变、跑批其实已经变了"这种谁都不报错的错位。
    """
    tmpl = (paths.ROOT / spec.prompt_template).read_text(encoding="utf-8")
    insts = list(pool) if spec.key is Config.C1 else adapt.subset_from_c1(pool, spec.n_instances)
    return adapt.prompt_fingerprint(
        insts, tmpl, demos,
        permutations=spec.reps if spec.option_permutation else 1)


def arms_params(arms: list[adapt.Arm]) -> dict:
    return {a.name: a.params for a in arms}


# --------------------------------------------------------------------------
# 门禁 1：单测
# --------------------------------------------------------------------------

_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected")


def _run_pytest(extra: list[str]) -> subprocess.CompletedProcess:
    """跑 pytest。

    ★ 必须显式清掉 pyproject 的 addopts：那里已经写了 `-q`，
      我们再传一次 `-q` 就成了 `-qq` —— 而 **`-qq` 不打印末尾计数行**。
      这正是门禁长期只能抓到"进度点行"、报告里看不出跑了多少用例的原因。
      `-o addopts=` 把配置里的 addopts 置空，由本函数显式控制输出详细度。
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-o", "addopts=", *extra],
        cwd=paths.ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def check_tests(g: Gate) -> None:
    """单测门禁。

    记录**两个**可核验的数：收集数 + 通过数。
    只记 `returncode` 是不够的 —— 门禁的价值在于留下能自证的证据。
    """
    col = _run_pytest(["--collect-only", "-q"])
    m = _COLLECTED_RE.search(col.stdout or "")
    collected = int(m.group(1)) if m else None

    proc = _run_pytest(["-q", "--tb=short"])
    pairs = _COUNT_RE.findall(proc.stdout or "")
    detail = " / ".join(f"{n} {k}" for n, k in pairs) or "(未解析到计数行)"
    if collected is not None:
        detail = f"收集 {collected} ｜ {detail}"

    if proc.returncode == 0:
        g.ok("单测", detail)
    else:
        # 失败时把短摘要一并留下，否则只看到一个计数不知道挂了什么
        tail = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("FAILED")]
        extra = ("；" + "；".join(tail[:3])) if tail else ""
        g.bad("单测", detail + extra)


# --------------------------------------------------------------------------
# 门禁 2：冻结指纹
# --------------------------------------------------------------------------

def load_frozen_meta() -> dict:
    p = paths.data() / FROZEN
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def compute_fingerprints(
    pool: list,
    matrix: list[adapt.ConfigSpec],
    dataset: str = "eval_set",
) -> tuple[dict, dict]:
    """算各配置的 prompt 指纹。

    `dataset` 决定 few-shot 取哪一组示例（C5b 按数据集给不同示例组）。
    门禁默认按 eval_set 算 —— 冻结表里存的也是 eval_set 的指纹。
    """
    fps, notes = {}, {}
    for spec in matrix:
        demos = None
        if spec.few_shot:
            vp = paths.data() / "valid_pool.jsonl"
            vpool = build_set.load_frozen(vp) if vp.exists() else []
            try:
                demos = adapt.load_demos(spec, vpool, dataset=dataset)
            except RuntimeError as e:
                notes[spec.key.value] = f"few-shot 示例池不可用：{e}"
                continue
        try:
            fps[spec.key.value] = prompt_fingerprint(pool, spec, demos)
        except Exception as e:  # noqa: BLE001
            notes[spec.key.value] = f"渲染失败：{e}"
    return fps, notes


# --------------------------------------------------------------------------
# 门禁 3：题集结构
# --------------------------------------------------------------------------

def check_dataset(g: Gate, pool: list) -> None:
    name = "题集结构"
    if not pool:
        g.bad(name, "eval_set.jsonl 为空")
        return

    ids = [i.id for i in pool]
    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    dup_q = [k for k, v in Counter(i.question for i in pool).items() if v > 1]

    bad: list[str] = []
    for i in pool:
        if len(i.options) < 2:
            bad.append(f"{i.id}: 选项数 {len(i.options)} < 2")
        if not i.gold_indices:
            bad.append(f"{i.id}: gold_indices 为空")
        elif any(not (0 <= gi < len(i.options)) for gi in i.gold_indices):
            bad.append(f"{i.id}: gold_indices 越界 {i.gold_indices} / {len(i.options)} 选项")
        if not isinstance(i.split, Split):
            bad.append(f"{i.id}: split 非法 {i.split}")
        if not isinstance(i.source, Source):
            bad.append(f"{i.id}: source 非法 {i.source}")

    if dup_ids:
        bad.append(f"重复 id {len(dup_ids)} 个：{dup_ids[:3]}")
    if bad:
        g.bad(name, "；".join(bad[:4]) + (f"（共 {len(bad)} 处）" if len(bad) > 4 else ""))
    else:
        g.ok(name, f"{len(pool)} 题，选项数 {min(len(i.options) for i in pool)}–"
                   f"{max(len(i.options) for i in pool)}，无越界")

    # 重复题干只是数据卫生问题，不影响结论正确性 —— 故为 WARN
    if dup_q:
        g.warn("题集重复题干", f"{len(dup_q)} 组题干完全相同（例：{dup_q[:2]}）")
    else:
        g.ok("题集重复题干", "无重复")

    # difficulty 全 unknown 会让分层结论失效 —— 明确报出来，但不阻断（C1 结论不依赖它）
    dif = Counter((i.stratum or {}).get("difficulty", "unknown") for i in pool)
    if set(dif) <= {"unknown"}:
        g.warn("难度分层可用性", f"difficulty 全部为 unknown（{len(pool)} 题）——"
                                "分层拆分中的 difficulty 轴不可用，报告不得据此下结论")
    else:
        g.ok("难度分层可用性", dict(dif))


# --------------------------------------------------------------------------
# 门禁 4/5/6/7/8：逐 run 校验产物
# --------------------------------------------------------------------------

def check_run(g: Gate, run_id: str, run_dir: Path, pool: list,
              matrix_by_key: dict, arms: list[adapt.Arm],
              frozen_meta: dict) -> None:
    mp = run_dir / "manifest.json"
    if not mp.exists():
        g.warn(f"[{run_id}] manifest", "无 manifest.json，跳过该 run 的产物校验")
        return
    man = json.loads(mp.read_text(encoding="utf-8"))
    smoke = is_smoke(run_id)

    # ---- 4. 数据哈希一致 ------------------------------------------------
    want = (frozen_meta.get("eval_set") or {}).get("sha256_16")
    got = man.get("data_hash")
    cname = f"[{run_id}] 数据哈希"
    declared = man.get("dataset", "eval_set")
    if smoke:
        g.skip(cname, f"演练 run（data_hash={got}），不参与全量比对")
    elif declared != "eval_set":
        # ★ 该 run 用的是**另一份题集**（如 C7 的 refusal_set）。
        #   拿它去比 eval_set 的哈希会把"用了别的数据集"误报成"题集被改了"。
        #   比"填个假哈希骗过门禁"诚实，也比"不写 manifest"可追溯。
        g.ok(cname, f"题集={declared}（非 eval_set，不与冻结哈希比对） hash={got}")
    elif want is None:
        g.warn(cname, "未冻结，无法比对（先跑 --freeze）")
    elif got != want:
        g.bad(cname, f"manifest={got} 当前题集={want} —— 题集已变，此 run 不可与其它 run 配对比较")
    else:
        g.ok(cname, got)

    # ---- 8. 参数冻结 ----------------------------------------------------
    # ★ 判据：这个 run 里的 manifest.arms 是**被测模型**还是**judge**？
    #   原先的假设是"judge 模型不在 arms.yaml 里，跳过"（下一行的 `arm not in cur`）。
    #   但 2026-10-05 把 `deepseek-v4-pro` 同时加成了**被测模型**（旗舰档参照）之后，
    #   该假设失效 ⇒ 所有用 v4-pro 当 judge 的老 run 都被误报参数漂移（7 项假失败）。
    #
    #   为什么不能靠 arms.yaml 区分：**同一个模型可以身兼两职**，
    #   而两个角色本来就用不同参数（judge 要短输出，被测模型要完整作答）。
    #   所以必须按 **run 的角色** 判断，而不是按模型名。
    #
    #   已验证的信号（全部 5 个 judge run 命中）：note 含 `judge=`，
    #   或配置里有 matrix 标为 judge 的项，或该 run 根本没有配置（审计/归因类）。
    _cfgs = man.get("configs") or []
    _is_judge_run = (
        "judge=" in (man.get("note") or "")
        or not _cfgs
        or any(getattr(matrix_by_key.get(Config(c)), "judge", False) for c in _cfgs)
    )
    pname = f"[{run_id}] 参数冻结"
    cur = arms_params(arms)
    ov = man.get("param_overrides") or {}
    drift = []
    for arm, p in (man.get("params_effective") or {}).items():
        if arm not in cur:
            continue                       # judge 模型不在 arms.yaml 里，跳过
        if _is_judge_run:
            continue                       # ★ 该 run 里它是 judge，参数不适用模型配置
        # ★ 有意的覆盖不算漂移：期望值 = arms.yaml + 本 run 声明的覆盖
        #   `param_arms`：覆盖**实际作用的模型**。用 `--param-arms` 限定后，
        #   未覆盖的模型必须按 arms.yaml 原值比对 —— 否则会把「有意只给部按模型
        #   加参数」误报成参数漂移（E1 的 longcat 就是这样被误报的）。
        #
        #   ★ `"<dropped>"` 是 `run_config.py --drop-param` 的哨兵：它表示该键被
        #     **删除**，而不是被设成这个字符串。不特判的话，期望值里会多出一个
        #     `thinking: "<dropped>"`，把「有意删除参数」误报成参数漂移
        #     （E1b 打开 longcat 思考就是这么触发的）。
        _pa = man.get("param_arms")
        _drop = {k for k, v in ov.items() if v == "<dropped>"}
        _ov = {k: v for k, v in ov.items() if k not in _drop}
        expect = {**cur[arm], **_ov} if (_pa is None or arm in _pa) else dict(cur[arm])
        for _k in _drop:
            expect.pop(_k, None)
        if norm(expect) != norm(p):
            drift.append(f"{arm}: manifest={norm(p)} vs 期望={norm(expect)}")
    if drift:
        g.bad(pname, "；".join(drift[:2]))
    elif _is_judge_run:
        g.ok(pname, f"judge run（{'无配置' if not _cfgs else _cfgs}）—— judge 参数不参与模型冻结检查")
    elif not (man.get("params_effective") or {}):
        g.warn(pname, "manifest 未记录 params_effective")
    elif ov:
        g.ok(pname, f"{len(man.get('params_effective'))} 个模型 = arms.yaml + 声明的覆盖 {ov}")
    else:
        g.ok(pname, f"{len(man.get('params_effective'))} 个模型与 arms.yaml 一致")

    # ---- 5/6. raw 与 tally 完整性 --------------------------------------
    n_arms = len(man.get("arms") or [])
    # ★ 题集必须按 **manifest 声明的 dataset** 取，不能一律用 eval_set。
    #   C-Eval 是 583 题、eval_set 是 588 题；用错会让完整的 run 被判"缺 25 条"。
    #   这类**假失败**比真失败更伤 —— 它训练人去忽略门禁。
    #
    # ⚠️ 但不是所有题集都能用 `load_frozen` 读：C7 的 `refusal_set.jsonl` /
    #    `orbench_set.jsonl` 是另一套结构（无 `split` 字段），且存在 run
    #    声明了一个**根本不存在**的 dataset 名（`or-bench`）。
    #    ⇒ 读不出来时**跳过该项检查并 warn**，绝不退回 eval_set ——
    #      退回就等于拿错题集的题数去比，又制造一批假失败。
    run_pool = pool
    ds_declared = man.get("dataset", "eval_set")
    pool_ok = True
    if ds_declared != "eval_set":
        ds_path = paths.data() / f"{ds_declared}.jsonl"
        if not ds_path.exists():
            g.warn(f"[{run_id}] 题集", f"manifest 声明 dataset={ds_declared}，但 data/{ds_declared}.jsonl 不存在 —— 跳过条数检查")
            pool_ok = False
        else:
            try:
                run_pool = build_set.load_frozen(ds_path)
            except Exception as e:  # noqa: BLE001
                g.warn(f"[{run_id}] 题集", f"dataset={ds_declared} 无法用 load_frozen 读取（{type(e).__name__}）—— 跳过条数检查")
                pool_ok = False
    for ckey in (man.get("configs") or []):
        try:
            spec = matrix_by_key[Config(ckey)]
        except KeyError:
            g.warn(f"[{run_id}/{ckey}] 配置", "matrix.yaml 中无此配置，跳过")
            continue
        if not pool_ok:
            continue                       # 题集读不出来 ⇒ 条数无从判断，跳过
        exp = expected_specs(run_pool, spec, n_arms, reps=man.get("reps"))

        # ---- judge run（C4）：记录数 = 题数 × **被判模型数** ----------------
        # 不能用 manifest.arms 推算 —— judge run 的 manifest 里只有 judge 自己一个模型，
        # 但它要分别评审三个模型的输出，实际记录数是 3 倍。
        if getattr(spec, "judge", False):
            results = execute.load_results(run_id, spec.key)
            n_inst = len(adapt.subset_from_c1(run_pool, spec.n_instances))
            targets = sorted({r.target_arm for r in results})
            missing = sum(1 for r in results if not r.target_arm)
            failed = [r for r in results if not r.success]
            exp_j = n_inst * len(targets)
            rname = f"[{run_id}/{ckey}] judge raw 完整性"
            if smoke:
                g.skip(rname, "演练 run")
            elif not _RAW_AVAILABLE:
                g.skip(rname, _RAW_HINT)
            elif missing:
                g.bad(rname, f"{missing} 条记录的 target_arm 为空 —— "
                             "三个模型的 judge 请求会互相覆盖（与 HANDOFF §4 坑 6 同类）")
            elif failed:
                g.bad(rname, f"{len(failed)} 条 success=False（例：{failed[0].error[:60]}）")
            elif len(targets) < 2:
                g.warn(rname, f"只评审了 {len(targets)} 个模型，无法按模型比较 self-preference")
            elif len(results) != exp_j:
                g.bad(rname, f"应有 {exp_j} 条（{n_inst} 题 × {len(targets)} 个模型），实有 {len(results)}")
            else:
                g.ok(rname, f"{len(results)}/{exp_j} 条全部成功，被判模型 {targets}")

            tname = f"[{run_id}/{ckey}] judge tally 完整性"
            jt = metrics.load_tallies(run_id, spec.key, drop_excluded=False)  # 计数校验需全量
            if smoke:
                g.skip(tname, "演练 run")
            elif not _RAW_AVAILABLE:
                # 应有的条数是「题数 × 被判模型数」，而被判模型只能从 raw 里读出来
                g.skip(tname, f"{_RAW_HINT}，无法核对应有条数（tally 实有 {len(jt)} 条）")
            elif len(jt) != len(results):
                g.bad(tname, f"应有 {len(results)} 条（judge 只有单口径），实有 {len(jt)}")
            else:
                g.ok(tname, f"{len(jt)} 条，被判模型 {sorted({t['arm'] for t in jt})}")
            continue

        rname = f"[{run_id}/{ckey}] raw 完整性"
        if smoke:
            g.skip(rname, f"演练 run（期望 {exp} 条）")
        elif not _RAW_AVAILABLE:
            g.skip(rname, f"{_RAW_HINT}；期望 {exp} 条")
        else:
            results = execute.load_results(run_id, spec.key)
            got_n = len(results)
            failed = [r for r in results if not r.success]
            if got_n != exp:
                g.bad(rname, f"应有 {exp} 条，实有 {got_n} 条（缺 {exp - got_n}）")
            elif failed:
                g.bad(rname, f"{len(failed)} 条 success=False（例："
                             f"{failed[0].arm}/{failed[0].instance_id} {failed[0].error}）")
            else:
                g.ok(rname, f"{got_n}/{exp} 条全部成功")

        tname = f"[{run_id}/{ckey}] tally 完整性"
        tallies = metrics.load_tallies(run_id, spec.key, drop_excluded=False)  # 计数校验需全量
        exp_t = exp * 2  # 双口径
        if smoke:
            g.skip(tname, f"演练 run（期望 {exp_t} 条）")
        elif len(tallies) != exp_t:
            g.bad(tname, f"应有 {exp_t} 条（{exp}×2 口径），实有 {len(tallies)} 条")
        else:
            modes = Counter(t.get("mode") for t in tallies)
            g.ok(tname, f"{len(tallies)} 条，口径分布 {dict(modes)}")

        # ---- 7. 判分口径不变量（核心断言）-----------------------------
        iname = f"[{run_id}/{ckey}] 二元判分零分歧"
        if not tallies:
            g.skip(iname, "无 tally")
            continue
        offenders = []
        for arm in sorted({t["arm"] for t in tallies}):
            b = metrics.judge_agreement_report(tallies, arm).get("binary", {})
            cm = b.get("confusion") or {}
            if cm.get("FN") or cm.get("FP"):
                offenders.append(f"{arm}: FN={cm.get('FN')} FP={cm.get('FP')}")
        if offenders:
            g.bad(iname, "严格/宽松在**判分结论**上出现分歧（不只是解析覆盖）：" + "；".join(offenders))
        else:
            g.ok(iname, "全部模型 FN=FP=0 —— 「口径不制造伪低分」成立")


# --------------------------------------------------------------------------
# 门禁 9：静默降级（声明了 few-shot 就不能跑成 0-shot）
# --------------------------------------------------------------------------

def check_silent_degradation(g: Gate, matrix: list[adapt.ConfigSpec]) -> None:
    for spec in matrix:
        if not spec.few_shot:
            continue
        name = f"[{spec.key.value}] few-shot 前置条件"
        vp = paths.data() / "valid_pool.jsonl"
        if not vp.exists():
            g.bad(name, "声明了 few-shot 但 data/valid_pool.jsonl 不存在 —— "
                        "若直接跑批会得到 0-shot 却被标成 few-shot（静默降级）")
            continue
        try:
            demos = adapt.load_demos(spec, build_set.load_frozen(vp), dataset="eval_set")
        except RuntimeError as e:
            g.bad(name, str(e))
            continue
        want = int((spec.few_shot or {}).get("n", 5))
        if len(demos) != want:
            g.bad(name, f"声明 {want} 条示例，实得 {len(demos)} 条")
        else:
            g.ok(name, f"{len(demos)} 条（取自 valid）")


# --------------------------------------------------------------------------
# 门禁 10：构造层纯函数
# --------------------------------------------------------------------------

def check_purity(g: Gate, pool: list, matrix: list[adapt.ConfigSpec],
                 arms: list[adapt.Arm]) -> None:
    name = "build_specs 纯函数"
    try:
        spec = next(s for s in matrix if s.key is Config.C1)
        adapt.assert_pure(
            adapt.build_specs, run_id="__purity__", config=spec, arms=arms, instances=pool[:20],
        )
        g.ok(name, "同输入两次调用结果一致（无运行时随机）")
    except AssertionError as e:
        g.bad(name, str(e))
    except Exception as e:  # noqa: BLE001
        g.warn(name, f"未能校验：{e}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="回归门禁")
    ap.add_argument("--skip-tests", action="store_true", help="跳过 pytest")
    ap.add_argument("--freeze", action="store_true", help="重新记录冻结指纹（题集/模板/参数改过后用）")
    ap.add_argument("--runs", default=None, help="逗号分隔的 run_id，默认自动发现 runs/*")
    ap.add_argument("--json", action="store_true", help="另写 reports/ci_gate.json")
    args = ap.parse_args()

    paths.reports()
    g = Gate()

    print("=" * 78)
    print("回归门禁")
    print("=" * 78)

    eval_path = paths.data() / "eval_set.jsonl"
    pool = build_set.load_frozen(eval_path) if eval_path.exists() else []
    _arms_cfg, arms = adapt.load_arms()
    matrix = adapt.load_matrix()
    matrix_by_key = {s.key: s for s in matrix}
    cur_hash = file_sha16(eval_path) if eval_path.exists() else None
    fps, fp_notes = compute_fingerprints(pool, matrix)
    frozen_meta = load_frozen_meta()

    # ---- --freeze：记录当前指纹 ----------------------------------------
    if args.freeze:
        meta = {
            "eval_set": {
                "file": "data/eval_set.jsonl",
                "sha256_16": cur_hash,
                "n_instances": len(pool),
                "frozen_at": datetime.now(timezone.utc).isoformat(),
            },
            "prompt_fingerprints": fps,
            "prompt_fingerprint_notes": fp_notes,
            "arms_params": arms_params(arms),
        }
        p = paths.data() / FROZEN
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已冻结指纹 -> {p.relative_to(paths.ROOT)}")
        print(f"  题集 {cur_hash}（{len(pool)} 题）")
        for k, v in sorted(fps.items()):
            print(f"  prompt[{k}] {v}")
        for k, v in sorted(fp_notes.items()):
            print(f"  [注意] prompt[{k}] 未记录：{v}")
        return 0

    # ---- 1. 单测 --------------------------------------------------------
    if args.skip_tests:
        g.skip("单测", "--skip-tests")
    else:
        check_tests(g)

    # ---- 2. 冻结指纹 ----------------------------------------------------
    fname = "冻结指纹"
    if not frozen_meta:
        g.bad(fname, f"未找到 data/{FROZEN} —— 先跑 --freeze 记录基线")
    else:
        fe = frozen_meta.get("eval_set") or {}
        if fe.get("sha256_16") != cur_hash:
            g.bad(fname, f"题集哈希变了：记录 {fe.get('sha256_16')} → 当前 {cur_hash}")
        else:
            g.ok(fname, f"题集 {cur_hash} 未变（{len(pool)} 题）")

        # prompt 指纹漂移
        drift = []
        for k, v in (frozen_meta.get("prompt_fingerprints") or {}).items():
            if k in fps and fps[k] != v:
                drift.append(f"{k}: {v} → {fps[k]}")
        if drift:
            g.bad("prompt 指纹", "模板或渲染逻辑已变，跨 run 配对比较失效：" + "；".join(drift)
                                + "（确认无误后跑 --freeze 重新记录）")
        else:
            g.ok("prompt 指纹", f"{len(frozen_meta.get('prompt_fingerprints') or {})} 个配置未变")

        # 参数漂移
        cur_p = arms_params(arms)
        old_p = frozen_meta.get("arms_params") or {}
        pdrift = [f"{a}: {norm(old_p.get(a))} → {norm(cur_p.get(a))}"
                  for a in cur_p if a in old_p and norm(cur_p[a]) != norm(old_p[a])]
        if pdrift:
            g.bad("模型参数指纹", "arms.yaml 参数已变：" + "；".join(pdrift)
                                + "（跑批参数必须冻结，改了要重跑）")
        else:
            g.ok("模型参数指纹", f"{len(cur_p)} 个模型未变")

    # ---- 3. 题集结构 ----------------------------------------------------
    check_dataset(g, pool)

    # ---- 4-8. 逐 run ----------------------------------------------------
    if args.runs:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        run_ids = sorted(p.name for p in (paths.ROOT / "runs").iterdir() if p.is_dir())
    for rid in run_ids:
        rd = paths.ROOT / "runs" / rid
        if rd.is_dir():
            check_run(g, rid, rd, pool, matrix_by_key, arms, frozen_meta)

    # ---- 9/10. 前置条件与纯函数 ----------------------------------------
    check_silent_degradation(g, matrix)
    check_purity(g, pool, matrix, arms)

    # ---- 汇总 ----------------------------------------------------------
    print()
    icon = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[--]  "}
    for r in g.results:
        line = f"{icon[r['status']]} {r['check']:<34}"
        if r["detail"]:
            line += f" {r['detail']}"
        print(line)

    n_fail = g.n_fail
    n_warn = sum(1 for r in g.results if r["status"] == WARN)
    print()
    print("-" * 78)
    print(f"共 {len(g.results)} 项：PASS {sum(1 for r in g.results if r['status'] == PASS)}｜"
          f"FAIL {n_fail}｜WARN {n_warn}｜SKIP {sum(1 for r in g.results if r['status'] == SKIP)}")
    print("门禁结果：" + ("不通过 —— 有结论级前提被破坏" if n_fail else "通过"))

    if args.json:
        out = paths.reports() / "ci_gate.json"
        out.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eval_hash": cur_hash,
            "n_fail": n_fail,
            "n_warn": n_warn,
            "results": g.results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告 -> {out.relative_to(paths.ROOT)}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
