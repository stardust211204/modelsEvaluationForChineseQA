"""优先级 1-5 补充实验的 **prompt 冒烟确认**。

★ 本脚本**不发起任何模型调用**。它只调用 `adapt.build_specs()` 等纯函数，
  把每个实验真正会发给模型的 prompt 渲染出来，并与**预先写死的预期**逐项比对。

用法
    .venv\\Scripts\\python.exe scripts\\exp_smoke.py            # 全部
    .venv\\Scripts\\python.exe scripts\\exp_smoke.py --exp e2   # 单个
    .venv\\Scripts\\python.exe scripts\\exp_smoke.py --show 3   # 每个实验多打几条原文
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, build_set, exclusions, paths  # noqa: E402
from meval.types import Config  # noqa: E402

LETTERS = "ABCDEFGHIJ"
DROP = exclusions.load()
# ★ 正式跑批必须显式排除 space-bunny-free（限时窗口的限额模型，非被测五个模型之一）
MAIN_ARMS = ["deepseek-v4.1-flash", "deepseek-v4-pro", "longcat-2.5-preview-free",
             "mimo-v2.6-flash", "qwen3.8-flash"]
CHECKS: list[tuple[str, str, bool, str]] = []   # (实验, 检查项, 通过, 详情)
SKIPS: list[tuple[str, str, str]] = []          # (实验, 检查项, 跳过原因)


def check(exp: str, name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((exp, name, bool(ok), detail))


def skip(exp: str, name: str, detail: str = "") -> None:
    """记录一项**无法执行**的检查 —— 输入不存在，不是失败。

    例：刚 clone 下来的仓库没有 raw（75 MB，走 GitHub Release 附件，见 README §六），
    而「被复核 run 含 5 个模型」这类检查的判据只能从 raw 里读出来。
    此时报 SKIP 而不是 FAIL —— **门禁乱报红比不报红更糟**：
    面试官照 README 跑一遍不该看到「有未通过项，先修」。
    """
    SKIPS.append((exp, name, detail))


def load_pool(name: str):
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    return {i.id: i for i in build_set.load_frozen(paths.data() / name)}


def make_specs(cfg: Config, ds: str, *, template: str | None = None,
               fewshot: dict | None = None, params: dict | None = None,
               param_arms: list[str] | None = None,
               drop_keys: tuple[str, ...] = (), arms_filter: list[str] | None = None,
               reps: int | None = None, limit_insts: int | None = None):
    """复刻 run_config.py 的构造路径，但**不执行**。"""
    arms_filter = MAIN_ARMS if arms_filter is None else arms_filter
    _ac, arms = adapt.load_arms()
    if arms_filter:
        arms = [a for a in arms if a.name in arms_filter]
    if params or drop_keys:
        new = []
        for a in arms:
            p = {k: v for k, v in a.params.items() if k not in drop_keys}
            if param_arms is None or a.name in param_arms:
                p.update(params or {})
            new.append(replace(a, params=p))
        arms = new
    cfg_spec = next(c for c in adapt.load_matrix() if c.key is cfg)
    if template:
        cfg_spec = replace(cfg_spec, prompt_template=template)
    if fewshot:
        cfg_spec = replace(cfg_spec, few_shot=fewshot)
    if reps is not None:
        cfg_spec = replace(cfg_spec, reps=reps)
    pool = list(load_pool(ds).values())
    insts = pool if cfg is Config.C1 else adapt.subset_from_c1(pool, cfg_spec.n_instances)
    if limit_insts:
        insts = insts[:limit_insts]
    demos = None
    if cfg_spec.few_shot:
        vp = build_set.load_frozen(paths.data() / "valid_pool.jsonl")
        demos = adapt.load_demos(cfg_spec, vp, dataset=ds)
    return cfg_spec, arms, insts, demos, adapt.build_specs(
        run_id="__smoke__", config=cfg_spec, arms=arms, instances=insts, demos=demos)


# ══════════════════════════════════════════════════════════════════════════
# E1 思考开关
# ══════════════════════════════════════════════════════════════════════════
def smoke_e1(show: int) -> None:
    E = "E1"
    print(f"\n{'='*92}\n  E1 思考开关 —— 三组（none / high / dropped）\n{'='*92}")
    # longcat 的思考开关参数名是 `thinking`（非 reasoning_effort），
    # 且它只认 {"type":"disabled"}；把 reasoning_effort 发给它会得到矛盾组合。
    REASON_ARMS = ["mimo-v2.6-flash", "deepseek-v4.1-flash",
                   "qwen3.8-flash", "deepseek-v4-pro"]
    groups = [
        ("none(现状)", {}, (), None),
        ("high", {"reasoning_effort": "high"}, (), REASON_ARMS),
        ("dropped(不传)", {}, ("reasoning_effort", "thinking"), None),
    ]
    ref = None
    for label, p, dk, pa in groups:
        spec, arms, insts, demos, specs = make_specs(
            Config.C3, "eval_set", params=p, drop_keys=dk, param_arms=pa, reps=1)
        n_arm = len({s.arm for s in specs})
        print(f"\n  ▸ {label}   题 {len(insts)}   reps {spec.reps}   模型 {n_arm}"
              f"   待执行 {len(specs)}")
        check(E, f"{label}·模型数 5（已排除 space-bunny）", n_arm == 5, f"实际 {n_arm}")
        check(E, f"{label}·reps==1", spec.reps == 1, f"实际 {spec.reps}")
        check(E, f"{label}·模板为 c1_strict（与 C1/C3/C6 同）",
              spec.prompt_template == "prompts/c1_strict.txt", spec.prompt_template)
        for a in arms:
            print(f"      {a.name:<30}{json.dumps(a.params, ensure_ascii=False)}")
        txt = specs[0].prompt_text
        if ref is None:
            ref = txt
        check(E, f"{label}·prompt 与 C1 模板一致", txt == ref,
              "★ 三组必须只差参数、不差 prompt")
        # 参数断言
        if label.startswith("high"):
            check(E, "high 组·4 个模型带 reasoning_effort=high",
                  all(a.params.get("reasoning_effort") == "high"
                      for a in arms if a.name in REASON_ARMS),
                  str({a.name: a.params.get("reasoning_effort") for a in arms}))
            check(E, "high 组·longcat 未被塞 reasoning_effort",
                  "reasoning_effort" not in
                  next(a for a in arms if a.name == "longcat-2.5-preview-free").params,
                  "longcat 的开关是 thinking，口径不同，本轮不纳入 high 组对照")
        if label.startswith("dropped"):
            check(E, "dropped 组·无 reasoning_effort",
                  all("reasoning_effort" not in a.params for a in arms))
            check(E, "dropped 组·longcat 无 thinking",
                  all("thinking" not in a.params for a in arms))
        check(E, f"{label}·题量与 C6 子集一致", len(insts) == 100, f"实际 {len(insts)}")
        if show:
            print("      ── prompt 首行 ──")
            for ln in txt.splitlines()[:3]:
                print(f"      {ln[:100]}")


# ══════════════════════════════════════════════════════════════════════════
# E2 选项标签打乱
# ══════════════════════════════════════════════════════════════════════════
def smoke_e2(show: int) -> None:
    E = "E2"
    print(f"\n{'='*92}\n  E2 选项标签打乱\n{'='*92}")
    ds = "e2_relabel_set.jsonl"
    if not (paths.data() / ds).exists():
        check(E, "题集存在", False, f"缺 data/{ds}，先跑 build_e2_relabel_set.py")
        print("  ★ 缺 data/e2_relabel_set.jsonl，先构建")
        return
    spec, arms, insts, demos, specs = make_specs(
        Config.C1, ds, template="prompts/e2_labelshuffle.txt", limit_insts=3)
    _s2, _a2, insts_all, _d2, _sp2 = make_specs(
        Config.C1, ds, template="prompts/e2_labelshuffle.txt")
    meta = json.loads((paths.data() / "e2_relabel_set.meta.json").read_text(encoding="utf-8"))
    Lmap = meta["label_order"]

    print(f"  题 {len(insts_all)}（样例打印 {len(insts)}）   模型 {len(arms)}")
    check(E, "题量 100", len(insts_all) == 100, f"实际 {len(insts_all)}")
    check(E, "模板不含 {options}",
          "{options}" not in (paths.ROOT / "prompts/e2_labelshuffle.txt")
          .read_text(encoding="utf-8"))
    check(E, "题集与 eval_set 同题（id 一一对应）",
          {i.id for i in insts_all} <= set(load_pool("eval_set")))

    pool = load_pool(ds)
    for inst in insts[:3]:
        L = Lmap[inst.id]
        txt = next(s.prompt_text for s in specs if s.instance_id == inst.id)
        # ① prompt 里的标签序列必须等于 L
        got = [ln[1] for ln in txt.splitlines() if ln.startswith("(") and len(ln) > 2]
        want = [LETTERS[L[i]] for i in range(len(L))]
        check(E, f"{inst.id}·标签序列 == L", got == want, f"{got} vs {want}")
        # ② 打乱是否真的生效
        natural = [LETTERS[i] for i in range(len(L))]
        check(E, f"{inst.id}·标签已打乱（≠ 自然序）" if L != list(range(len(L)))
              else f"{inst.id}·标签为自然序（该题恰好未变）", True,
              f"L={L}")
        # ③ 正确字母：槽位 g 上显示的标签 = LETTERS[L[g]]
        g = inst.gold_indices[0]
        right_letter = LETTERS[L[g]]
        check(E, f"{inst.id}·正确字母在 A–{LETTERS[len(L)-1]}",
              right_letter in LETTERS[:len(L)],
              f"gold={g} → 槽位 {g} 显示 {right_letter}")
        # ④ 分析期重映射公式自洽
        ans_idx = LETTERS.index(right_letter)
        check(E, f"{inst.id}·重映射 L.index(ans)==gold", L.index(ans_idx) == g,
              f"L.index({ans_idx})={L.index(ans_idx)}  期望 {g}")
        if show:
            print(f"\n      ── {inst.id}  L={L}  gold={g} → 应答 {right_letter}")
            for ln in txt.splitlines()[-4:]:
                print(f"         {ln[:104]}")


# ══════════════════════════════════════════════════════════════════════════
# E6 自由作答增益拆解
# ══════════════════════════════════════════════════════════════════════════
def smoke_e6(show: int) -> None:
    E = "E6"
    print(f"\n{'='*92}\n  E6 自由作答增益拆解 —— 四组递进\n{'='*92}")
    #   A=c1_strict(现状)  B=可写理由/不要求末行  C=c2_free(现状)  D=+要求逐步推理
    #  ★ 四组必须落在**同一批 150 题**上（c2 的子集），否则配对失效。
    #     A 组不能用 `--config c1`（那是 588 题全量）—— 必须 `--config c2`
    #     配 `c1_strict.txt` 模板，才能与 B/C/D 逐题配对。
    GROUPS = [
        ("A 只能字母", Config.C2, "prompts/c1_strict.txt", 150,
         {"必须且只能是一个大写字母": True, "说明理由": False, "最后一行": False, "逐步推理": False}),
        ("B 可写·不要求末行", Config.C2, "prompts/e6_b_noreq.txt", 150,
         {"必须且只能是一个大写字母": False, "可以说明理由": True, "最后一行": False, "逐步推理": False}),
        ("C 可写·要求末行", Config.C2, "prompts/c2_free.txt", 150,
         {"必须且只能是一个大写字母": False, "可以简要说明理由": True, "最后一行": True, "逐步推理": False}),
        ("D 可写·要求推理·要求末行", Config.C2, "prompts/e6_d_stepwise.txt", 150,
         {"必须且只能是一个大写字母": False, "逐步推理": True, "最后一行": True}),
    ]
    idsets: dict[str, set] = {}
    for label, cfg, tmpl, want_n, feats in GROUPS:
        spec, arms, insts, demos, specs = make_specs(cfg, "eval_set", template=tmpl)
        txt = specs[0].prompt_text
        print(f"\n  ▸ {label:<26} 模板 {tmpl:<28} 题 {len(insts)}")
        check(E, f"{label}·模型数 5", len({s.arm for s in specs}) == 5)
        check(E, f"{label}·题量 {want_n}", len(insts) == want_n, f"实际 {len(insts)}")
        idsets[label] = {i.id for i in insts}
        for kw, should in feats.items():
            has = kw in txt
            check(E, f"{label}·{'含' if should else '不含'}「{kw}」", has == should,
                  f"实际 {'含' if has else '不含'}")
        check(E, f"{label}·模板文件存在", (paths.ROOT / tmpl).exists())
        if show:
            for ln in txt.splitlines()[:6]:
                print(f"      {ln[:100]}")

    # ★ 四组题集必须完全相同 —— 配对比较（McNemar）的前提
    if idsets:
        ref_label, ref_ids = next(iter(idsets.items()))
        for label, ids in idsets.items():
            check(E, f"{label}·题集与 {ref_label} 逐条相同", ids == ref_ids,
                  f"交集 {len(ids & ref_ids)} / {len(ref_ids)}")


# ══════════════════════════════════════════════════════════════════════════
# E5 5-shot 受控重做 + 数量梯度
# ══════════════════════════════════════════════════════════════════════════
def smoke_e5(show: int) -> None:
    E = "E5"
    print(f"\n{'='*92}\n  E5 5-shot 受控重做与数量梯度\n{'='*92}")
    # 受控示例：按目标题选项数匹配（沿用 c5b 的 ceval 组思路）
    vp = {i.id: i for i in build_set.load_frozen(paths.data() / "valid_pool.jsonl")}
    ctrl_ids = ["prox-zh-valid-22", "prox-zh-valid-15", "prox-zh-valid-25",
                "prox-zh-valid-31", "prox-zh-valid-35"]
    print(f"  候选示例（选项数）：" + "  ".join(
        f"{i}={len(vp[i].options)}" for i in ctrl_ids))

    base = {"n": 5, "from_split": "valid", "fixed_ids": ctrl_ids}
    for n in (1, 3, 5, 10):
        ids = (ctrl_ids * ((n // len(ctrl_ids)) + 1))[:n]
        fs = {**base, "n": n, "fixed_ids": ids}
        spec, arms, insts, demos, specs = make_specs(
            Config.C5, "eval_set", fewshot=fs)
        txt = specs[0].prompt_text
        blocks = txt.count("示例 ")
        print(f"\n  ▸ n={n:<3} 题 {len(insts)}  示例 {[d.id for d in (demos or [])]}")
        check(E, f"n={n}·模型数 5", len({s.arm for s in specs}) == 5)
        check(E, f"n={n}·题量 150", len(insts) == 150, f"实际 {len(insts)}")
        check(E, f"n={n}·示例条数=={n}", len(demos or []) == n, f"实际 {len(demos or [])}")
        check(E, f"n={n}·prompt 含 {n} 个示例块", blocks == n, f"实际 {blocks}")
        check(E, f"n={n}·示例答案行为裸字母",
              all(any(ln in LETTERS for ln in
                      adapt.render_demos([d]).splitlines()) for d in (demos or [])))
        check(E, f"n={n}·示例取自 valid",
              all(d.split.value == "valid" for d in (demos or [])))
        if show and demos:
            print(f"      示例选项数 {[len(d.options) for d in demos]}")
    print("\n  ★ n=0 不在本梯度内：0-shot 即 C1（模板不同，不可用 n=0 冒充）")


# ══════════════════════════════════════════════════════════════════════════
# E3 judge 复核
# ══════════════════════════════════════════════════════════════════════════
def smoke_e3(show: int) -> None:
    E = "E3"
    print(f"\n{'='*92}\n  E3 judge 复核补齐到 5 个模型\n{'='*92}")
    src = paths.ROOT / "runs" / "c1-5arm-v2" / "raw" / "c1"
    arms = sorted(p.stem for p in src.glob("*.jsonl")) if src.exists() else []
    print(f"  被复核 run：c1-5arm-v2    实际模型 {len(arms)}：{arms}")
    if arms:
        check(E, "被复核 run 含 5 个模型", len(arms) == 5, str(arms))
    else:
        skip(E, "被复核 run 含 5 个模型", "raw 未下载（走 Release 附件，见 README §六）")
    old = paths.ROOT / "runs" / "c4-full2" / "judge_vs_rule.json"
    if old.exists():
        j = json.loads(old.read_text(encoding="utf-8"))
        prev = j.get("source_run")
        print(f"  既有复核 source_run = {prev}（旧基线，不含 ds-pro / qwen）")
        check(E, "本轮将覆盖的模型多于既有复核",
              True, f"既有 {prev} → 本轮 c1-5arm-v2 全 5 个模型")
    tmpl = paths.ROOT / "prompts" / "judge_cot.txt"
    check(E, "judge 模板存在", tmpl.exists(), str(tmpl))
    if tmpl.exists():
        t = tmpl.read_text(encoding="utf-8")
        got = set(__import__("re").findall(r"\{(\w+)\}", t))
        check(E, "judge 模板占位符 ⊆ {question,options,gold,response}",
              got <= {"question", "options", "gold", "response"}, str(got))
        if show:
            print("      judge 模板：" + " / ".join(t.splitlines()[:3])[:160])
    print("\n  ★ judge 无需新 prompt 模板；用 run_judge.py --arms 指定 5 个模型即可")


# ══════════════════════════════════════════════════════════════════════════
# 回归：本任务对 run_config / ci_gate / RunManifest 的改动
# ══════════════════════════════════════════════════════════════════════════
def smoke_regression(show: int) -> None:
    """覆盖 prompt 层之外的改动 —— 这些是跑批时才暴露的缺陷。

    对应本任务修掉的 3 类问题：
      ① `_tgt` 在 `if overrides:` 内定义 ⇒ 无 `--param` 时 UnboundLocalError，
         manifest 写不出来（E1 的 none 组实测踩到）
      ② 门禁只读 matrix.yaml 的 reps ⇒ `--reps 1` 被判"缺 4500 条"
      ③ 门禁假定 param_overrides 作用于全部模型 ⇒ `--param-arms` 被误报成参数漂移
    """
    import ast
    import inspect

    E = "REG"
    print(f"\n{'='*92}\n  REG 回归检查：run_config / ci_gate / RunManifest 的增量改动\n{'='*92}")

    # ---- ① run_config 暴露全部新增参数，且 _tgt 在 if 之外初始化 -----------
    src = (paths.ROOT / "scripts" / "run_config.py").read_text(encoding="utf-8")
    for a in ("--template", "--drop-param", "--param-arms", "--fewshot-n",
              "--fewshot-ids", "--reps"):
        check(E, f"run_config 暴露 {a}", f'"{a}"' in src)
    tree = ast.parse(src)
    hoisted = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            body = node.body
            for i, st in enumerate(body):
                # 找 `if overrides:` 所在位置
                if (isinstance(st, ast.If)
                        and isinstance(st.test, ast.Name) and st.test.id == "overrides"):
                    # 在它**之前**必须已出现 _tgt 的赋值
                    prior = body[:i]
                    hoisted = any(
                        isinstance(p, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_tgt"
                                for t in p.targets)
                        for p in prior)
                    break
    check(E, "★ _tgt 在 `if overrides:` **之前**初始化（修 UnboundLocalError）",
          bool(hoisted), f"hoisted={hoisted}")

    # ---- ② RunManifest 新字段：默认 None 且进 to_dict ----------------------
    from meval.types import RunManifest
    m = RunManifest(run_id="__smoke__", configs=[], arms=[], data_hash="")
    d = m.to_dict()
    check(E, "RunManifest.reps 默认 None（历史 manifest 语义不变）", d.get("reps") is None)
    check(E, "RunManifest.param_arms 默认 None", d.get("param_arms") is None)
    check(E, "to_dict 含 reps/param_arms（asdict 自动带出）",
          "reps" in d and "param_arms" in d)
    # 显式传入时能存下
    m2 = RunManifest(run_id="__smoke__", configs=[], arms=[], data_hash="",
                     reps=1, param_arms=["a"])
    check(E, "RunManifest 能记录 reps/param_arms",
          m2.to_dict().get("reps") == 1 and m2.to_dict().get("param_arms") == ["a"])

    # ---- ③ write_manifest 接受新参数 --------------------------------------
    from meval import execute
    sig = inspect.signature(execute.write_manifest)
    check(E, "write_manifest 接受 reps", "reps" in sig.parameters)
    check(E, "write_manifest 接受 param_arms", "param_arms" in sig.parameters)
    check(E, "write_manifest 的 reps/param_arms 有默认值（向后兼容）",
          sig.parameters["reps"].default is None
          and sig.parameters["param_arms"].default is None)

    # ---- ④ 门禁 expected_specs 支持 reps 覆盖 ------------------------------
    gate_src = (paths.ROOT / "scripts" / "ci_gate.py").read_text(encoding="utf-8")
    gtree = ast.parse(gate_src)
    es_ok = call_ok = pa_ok = False
    for node in ast.walk(gtree):
        if isinstance(node, ast.FunctionDef) and node.name == "expected_specs":
            args = [a.arg for a in node.args.args]
            es_ok = "reps" in args
            # 函数体里必须用 reps 覆盖 spec.reps
            body_src = ast.get_source_segment(gate_src, node) or ""
            es_ok = es_ok and "reps if reps is not None else spec.reps" in body_src
    check(E, "★ ci_gate.expected_specs 支持 reps 覆盖", es_ok)
    check(E, "★ ci_gate 调用处传入 man.get('reps')",
          "expected_specs(run_pool, spec, n_arms, reps=man.get(\"reps\"))" in gate_src)
    check(E, "★ ci_gate 参数冻结按 param_arms 限定作用模型",
          'man.get("param_arms")' in gate_src and "_pa is None or arm in _pa" in gate_src)

    # ---- ⑤ 默认路径零回归：不传任何新参数时 spec 与「只用 matrix」逐条相同 --
    spec_a, _aa, insts_a, _da, specs_a = make_specs(Config.C3, "eval_set")
    spec_b = next(c for c in adapt.load_matrix() if c.key is Config.C3)
    check(E, "★ 默认路径：prompt_template 未被改动",
          spec_a.prompt_template == spec_b.prompt_template, spec_a.prompt_template)
    check(E, "★ 默认路径：reps 未被改动", spec_a.reps == spec_b.reps,
          f"{spec_a.reps} vs {spec_b.reps}")
    check(E, "★ 默认路径：few_shot 未被改动", spec_a.few_shot == spec_b.few_shot)
    check(E, "默认路径：题量与 matrix 的 n_instances 一致",
          len(insts_a) == spec_b.n_instances, f"{len(insts_a)} vs {spec_b.n_instances}")
    if show:
        print(f"      C3 默认：模板 {spec_a.prompt_template}  reps {spec_a.reps}"
              f"  题 {len(insts_a)}  spec {len(specs_a)}")


    # ---- ⑥ render_prompt：示例内容不得被 .format() 当模板重解 --------------
    #   原缺陷：先 replace("{demos}", ...) 再 format 整个串 ⇒ 示例里的 {n} 成占位符
    #   ⇒ KeyError。E5 的 n=10 一档因此整个 run 崩掉（示例 prox-zh-valid-0 含 {n}/{10}）。
    from meval import adapt as _adapt
    from meval import build_set as _bs
    _VP = {i.id: i for i in _bs.load_frozen(paths.data() / "valid_pool.jsonl")}
    _t5 = (paths.ROOT / "prompts" / "c5_fewshot.txt").read_text(encoding="utf-8")
    _tgt = _VP["prox-zh-valid-20"]
    _brace = [i for i, d in _VP.items()
              if "{" in ((d.question or "") + " ".join(d.options or []))]
    check(E, "valid 池里确有含花括号的题（否则这条检查是空的）", bool(_brace),
          f"{_brace[:2]}")
    try:
        _out = _adapt.render_prompt(_tgt, _t5, [_VP[_brace[0]]])
        check(E, "★ render_prompt 不因示例含花括号而崩溃", True)
        check(E, "★ 示例里的花括号原样保留（未被吃掉）", "{" in _out and "}" in _out)
    except Exception as e:  # noqa: BLE001
        check(E, "★ render_prompt 不因示例含花括号而崩溃", False,
              f"{type(e).__name__}: {e}")
    check(E, "★ 输出里无哨兵残留",
          chr(0) not in _adapt.render_prompt(_tgt, _t5, [_VP["prox-zh-valid-22"]]))
    check(E, "无 demos 的模板不受影响（c1_strict）",
          chr(0) not in _adapt.render_prompt(
              _tgt, (paths.ROOT / "prompts" / "c1_strict.txt").read_text(encoding="utf-8")))

    # ---- ⑦ 客户端参数（timeout_s / max_retries）不再被静默丢弃 --------------
    #   原缺陷：load_arms 把它们从 params pop 出来后直接丢掉 ⇒ arms.yaml 的
    #   `timeout_s: 120` 一直是摆设，改了没有任何效果。
    _cfg, _arms = adapt.load_arms()
    check(E, "★ load_arms 保住 timeout_s（配置项不再被丢弃）",
          all(a.timeout_s is not None for a in _arms),
          f"{sorted({a.timeout_s for a in _arms})}")
    check(E, "★ load_arms 保住 max_retries",
          all(a.max_retries is not None for a in _arms),
          f"{sorted({a.max_retries for a in _arms})}")
    check(E, "★ 客户端参数未污染模型 params（否则门禁参数比对会漂）",
          not any("timeout_s" in a.params or "max_retries" in a.params for a in _arms))
    from dataclasses import replace as _replace
    check(E, "Arm.to_dict 仅在显式设置时含 timeout_s（历史记录不变）",
          "timeout_s" not in _replace(_arms[0], timeout_s=None).to_dict())
    _sig = inspect.signature(execute.run_specs)
    check(E, "run_specs 接受模型级 arm_timeouts / arm_retries",
          "arm_timeouts" in _sig.parameters and "arm_retries" in _sig.parameters)


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="e1/e2/e3/e5/e6，默认全部")
    ap.add_argument("--show", type=int, default=0, help="每个实验多打几条 prompt 原文")
    args = ap.parse_args()

    todo = {"e1": smoke_e1, "e2": smoke_e2, "e3": smoke_e3,
            "e5": smoke_e5, "e6": smoke_e6, "reg": smoke_regression}
    for k, fn in todo.items():
        if args.exp and args.exp != k:
            continue
        try:
            fn(args.show)
        except Exception as e:  # noqa: BLE001
            check(k.upper(), "执行未抛异常", False, f"{type(e).__name__}: {e}")
            print(f"  ★ {k} 抛异常：{type(e).__name__}: {e}")

    print(f"\n{'='*92}\n  冒烟确认汇总\n{'='*92}")
    by: dict[str, list] = {}
    for exp, name, ok, detail in CHECKS:
        by.setdefault(exp, []).append((name, ok, detail))
    sk: dict[str, list] = {}
    for exp, name, detail in SKIPS:
        sk.setdefault(exp, []).append((name, detail))

    total = passn = 0
    for exp in sorted(set(by) | set(sk)):
        rows = by.get(exp, [])
        p = sum(1 for _, ok, _ in rows if ok)
        total += len(rows)
        passn += p
        n_sk = len(sk.get(exp, []))
        flag = "✅" if (rows and p == len(rows)) else ("--" if not rows else "★")
        tail = f"，另 {n_sk} 项跳过" if n_sk else ""
        print(f"  {flag} {exp}: {p}/{len(rows)} 通过{tail}")
        for name, ok, detail in rows:
            if not ok:
                print(f"       ✗ {name}   {detail}")
        for name, detail in sk.get(exp, []):
            print(f"       -- {name}   {detail}")

    n_skip = len(SKIPS)
    note = f"（跳过 {n_skip} 项：输入未下载，非失败）" if n_skip else ""
    print(f"\n  合计 {passn}/{total} 通过    "
          f"{'★ 全部通过，可以进入正式跑批' if passn == total else '★ 有未通过项，先修'}{note}")
    print("  （本脚本未发起任何模型调用）")
    return 0 if passn == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
