"""单配置跑批编排：读冻结题集 → 构造 → 执行 →（双口径）判分。

用法：
    # 缩量试跑（只跑前 30 条，验证并发与错误率）
    .venv\\Scripts\\python.exe scripts\\run_config.py --config c1 --limit 30

    # 全量 C1（588 题 × 3 个模型 = 1,764 条）
    .venv\\Scripts\\python.exe scripts\\run_config.py --config c1

    # 断点续跑：同一 --run-id 再跑一次即可（已完成的会跳过）
    .venv\\Scripts\\python.exe scripts\\run_config.py --config c1 --run-id c1-20261003

产物：
    runs/{run_id}/manifest.json
    runs/{run_id}/raw/{config}/{arm}.jsonl
    runs/{run_id}/tallies/{config}/{arm}.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))

from meval import adapt, build_set, execute, metrics, paths, tally  # noqa: E402
from meval.types import Config, ParseMode, Tier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c1", help="配置键：c1/c2/c3/c5")
    ap.add_argument("--run-id", default=None, help="不指定则新建")
    ap.add_argument("--limit", type=int, default=None, help="本次最多执行多少条（试跑用）")
    ap.add_argument("--arms", default=None, help="逗号分隔的模型名，默认全部")
    ap.add_argument("--no-tally", action="store_true", help="只跑批不判分")
    ap.add_argument("--dataset", default="eval_set",
                    help="题集名（data/<名>.jsonl）。默认 eval_set；跨题库验证用 ceval_set"),
    ap.add_argument("--param", action="append", default=None,
                    help="覆盖模型参数（可重复）：--param max_tokens=2048")
    # ---- 以下四项为**补充验证实验**新增，默认行为与新增前逐字节相同 ----------
    #   设计前提：`configs/matrix.yaml` 与 `src/meval/*` 一律不改动。
    #   门禁 `ci_gate.py` 会 `compute_fingerprints(pool, matrix)`，往 matrix.yaml
    #   里加配置块会改变指纹集合、直接判 FAIL —— 所以实验配置只能走命令行覆盖。
    ap.add_argument("--template", default=None,
                    help="覆盖 prompt 模板路径（如 prompts/e6_b_noreq.txt）")
    ap.add_argument("--drop-param", action="append", default=None,
                    help="从全部模型的参数里删除某个键（可重复）：--drop-param reasoning_effort")
    ap.add_argument("--fewshot-n", type=int, default=None,
                    help="覆盖 few_shot.n（示例数量梯度实验用）")
    ap.add_argument("--fewshot-ids", default=None,
                    help="覆盖 few_shot.fixed_ids，逗号分隔（受控 few-shot 用）")
    ap.add_argument("--param-arms", default=None,
                    help="限定 --param 只作用于这些模型（逗号分隔），默认全部。"
                         "E1 用：longcat 的思考开关是 `thinking` 而非 "
                         "`reasoning_effort`，不能把后者发给它")
    ap.add_argument("--reps", type=int, default=None,
                    help="覆盖 reps。E1 用 `--config c3 --reps 1` 得到"
                         "「100 题子集 + reps=1 + c1 模板」，矩阵里没有这个组合")
    ap.add_argument("--timeout", type=float, default=None,
                    help="客户端超时（秒）。默认不传 ⇒ 用 ChatClient 默认 120s。"
                         "开思考的请求实测会超过 120s，需调大")
    args = ap.parse_args()

    cfg_key = Config(args.config)
    run_id = args.run_id or f"{args.config}-{time.strftime('%Y%m%d-%H%M%S')}"

    print("=" * 76)
    print(f"配置 {cfg_key.value}    run_id = {run_id}")
    print("=" * 76)

    # ---- 载入配置与题集 ----------------------------------------------------
    _arms_cfg, arms = adapt.load_arms()
    matrix = adapt.load_matrix()
    spec_cfg = next((c for c in matrix if c.key is cfg_key), None)
    if spec_cfg is None:
        print(f"!! matrix.yaml 中没有配置 {cfg_key.value}")
        return 2

    # ---- 补充验证实验的配置覆盖（全部走命令行，不改 matrix.yaml）----------
    #   默认（不传这四项）时 spec_cfg 与改动前**完全相同**，历史 run 不受影响。
    if args.template:
        tpath = Path(args.template)
        if not tpath.is_absolute():
            tpath = paths.ROOT / tpath
        if not tpath.exists():
            print(f"!! --template 指向的文件不存在：{tpath}")
            return 2
        spec_cfg = replace(spec_cfg, prompt_template=args.template)
        print(f"模板覆盖    : {args.template}")
    if args.fewshot_n is not None or args.fewshot_ids is not None:
        fs = dict(spec_cfg.few_shot or {})
        if args.fewshot_n is not None:
            fs["n"] = args.fewshot_n
        if args.fewshot_ids is not None:
            fs["fixed_ids"] = [x.strip() for x in args.fewshot_ids.split(",") if x.strip()]
            fs.pop("fixed_ids_by_dataset", None)   # 显式指定时以 fixed_ids 为准
        spec_cfg = replace(spec_cfg, few_shot=fs)
        print(f"few-shot 覆盖: n={fs.get('n')} ids={fs.get('fixed_ids')}")
    if args.reps is not None:
        spec_cfg = replace(spec_cfg, reps=args.reps)
        print(f"reps 覆盖   : {args.reps}")

    # ---- 题集选择 ----------------------------------------------------------
    # ★ 默认仍是 eval_set（MMLU-ProX-zh），**这不是可选项而是纪律**：
    #   `data/eval_set.jsonl` 是冻结哈希的基准，改它会让所有历史 run 的 manifest
    #   判为"不可配对"。所以跨题库验证走 `--dataset ceval_set`，
    #   生成一份**独立的** run，两边各自保持哈希一致。
    #   manifest 会记下 dataset 名，门禁据此跳过非 eval_set 的哈希比对
    #   （见 ci_gate.py 的 "该 run 用的是另一份题集" 分支）。
    ds_name = args.dataset
    eval_path = paths.data() / f"{ds_name}.jsonl"
    if not eval_path.exists():
        print(f"!! 未找到 data/{ds_name}.jsonl"
              + ("，请先运行 scripts/build_set.py" if ds_name == "eval_set"
                 else f"（可用题集：{[p.stem for p in paths.data().glob('*_set.jsonl')]}）"))
        return 2
    pool = build_set.load_frozen(eval_path)
    if ds_name != "eval_set":
        print(f"题集        : {ds_name}（{len(pool)} 题，非 eval_set，不与冻结哈希比对）")

    arm_filter = [a.strip() for a in args.arms.split(",")] if args.arms else None

    # ---- ★ 模型集合冻结：续跑时以 manifest 记录的模型为准 ----------------------
    # 修的是一个**静默**故障：`arms.yaml` 每次调用都重读，所以批次跑到一半时
    # 往 arms.yaml 加一个模型，**后续 run 会悄悄多跑一个模型**。
    # 实测发生过：给限时窗口的代号模型加模型后，正在排队的 `c3` 变成了 6 个模型。
    # 后果不是报错，而是"同名的 run 里模型集合不一样"—— 分析时按 run 取数就会串。
    #
    # 处置原则：**已有的模型记录是权威**。续跑时默认沿用 manifest 里的模型集合，
    # 只有显式传 `--arms` 才能改。这样中途改 arms.yaml 对进行中的批次无害。
    recorded: list[str] | None = None
    mp_existing = paths.manifest_path(args.run_id) if args.run_id else None
    if mp_existing and mp_existing.exists():
        try:
            recorded = json.loads(mp_existing.read_text(encoding="utf-8")).get("arms") or None
        except Exception:  # noqa: BLE001
            recorded = None
    if recorded and arm_filter is None:
        by_name = {a.name: a for a in arms}
        missing = [n for n in recorded if n not in by_name]
        if missing:
            print(f"!! 该 run 记录的模型 {missing} 已不在 arms.yaml 中 —— 无法续跑")
            return 2
        selected = [by_name[n] for n in recorded]
        print(f"模型集合      : 沿用该 run 记录的 {len(selected)} 个模型（忽略 arms.yaml 的后续改动）")
        extra = [a.name for a in arms if a.name not in recorded]
        if extra:
            print(f"              ⚠️ arms.yaml 里另有 {extra}，**本次不会跑**"
                  f"（如需加入请显式传 --arms）")
    else:
        selected = [a for a in arms if not arm_filter or a.name in arm_filter]
        if recorded and arm_filter is not None:
            added = [a.name for a in selected if a.name not in recorded]
            if added:
                print(f"              ⚠️ --arms 显式引入了该 run 原本没有的模型 {added}"
                      f" —— 它会被并入 manifest，请注意可比性")

    # ---- 显式参数覆盖（稳健性重跑用，如 --param max_tokens=2048）-----------
    # ★ 会记进 manifest.param_overrides：门禁据此区分
    #   「有意的参数变体」与「参数悄悄漂移」。没有这一步，
    #   任何合法的变体重跑都会被门禁误报成漂移。
    overrides: dict = {}
    for kv in (args.param or []):
        if "=" not in kv:
            print(f"!! --param 需要 KEY=VALUE 形式，收到：{kv}")
            return 2
        k, _, v = kv.partition("=")
        overrides[k.strip()] = _coerce(v.strip())
    #   ★ `_tgt` 必须在 if **之外**初始化：没有 `--param` 时（如 E1 的 none 组）
    #     它只在 if 内定义会 UnboundLocalError，manifest 写不出来 ——
    #     实测第 1 组的 manifest 因此缺 reps/param_arms，门禁无法判断条数。
    _tgt = ({x.strip() for x in args.param_arms.split(",")}
            if args.param_arms else None)
    if overrides:
        selected = [replace(a, params={**a.params, **overrides})
                    if (_tgt is None or a.name in _tgt) else a
                    for a in selected]
        print(f"参数覆盖    : {overrides}"
              + (f"   仅作用于 {sorted(_tgt)}" if _tgt else ""))

    # ---- --drop-param：从全部模型删除某个参数键（补充验证实验用）-------------
    #   动机：`--param` 只能**设置**键，无法**删除**。而 E1「思考开关」的第三组
    #   要测的是「完全不传思考参数时端点的默认行为」—— 必须真的把键去掉。
    #   删除同样记进 manifest.param_overrides（值记成 "<dropped>"），
    #   门禁据此区分「有意的参数变体」与「参数悄悄漂移」。
    if args.drop_param:
        for k in args.drop_param:
            selected = [replace(a, params={kk: vv for kk, vv in a.params.items() if kk != k})
                        for a in selected]
            overrides[k] = "<dropped>"
        print(f"参数删除    : {args.drop_param}")

    # ---- 题集范围：C1 全量；其余配置取 C1 子集 -----------------------------
    if cfg_key is Config.C1:
        instances = pool
    else:
        instances = adapt.subset_from_c1(pool, spec_cfg.n_instances)
    print(f"题集        : {len(instances)} 题"
          + ("" if cfg_key is Config.C1 else f"（C1 子集，目标 {spec_cfg.n_instances}）"))
    print(f"模型          : {', '.join(a.name for a in selected)}")

    # ---- few-shot（仅 C5）-------------------------------------------------
    demos = None
    if spec_cfg.few_shot:
        valid_path = paths.data() / "valid_pool.jsonl"
        vpool = build_set.load_frozen(valid_path) if valid_path.exists() else []
        demos = adapt.load_demos(spec_cfg, vpool, dataset=ds_name)
        print(f"few-shot    : {len(demos)} 条示例（取自 valid）")

    # ---- 构造 --------------------------------------------------------------
    specs = adapt.build_specs(
        run_id=run_id, config=spec_cfg, arms=selected,
        instances=instances, demos=demos,
    )
    print(f"待执行      : {len(specs)} 条")

    # ---- 模板变更守卫 ------------------------------------------------------
    # ★ 断点续跑的键是 (config, arm, instance_id, rep)，**不含 prompt**。
    #   改了模板却沿用同一 run_id，scan_done() 会把全部请求当成"已完成"跳过，
    #   于是产出一批用**旧模板**跑出来的数据 —— 而且不报任何错。
    #   （C5 首轮作废后重跑时就差一点踩到：见 docs/open_issues.md P1-6。）
    tmpl_text = (paths.ROOT / spec_cfg.prompt_template).read_text(encoding="utf-8")
    fp = adapt.prompt_fingerprint(
        instances, tmpl_text, demos,
        permutations=spec_cfg.reps if spec_cfg.option_permutation else 1)
    mpath = paths.manifest_path(run_id)
    if mpath.exists():
        try:
            old = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
        old_fp = (old.get("prompt_fingerprints") or {}).get(cfg_key.value)
        if old_fp and old_fp != fp:
            print(f"\n!! 拒跑：run_id={run_id} 已存在，但其记录的 prompt 指纹与当前不一致")
            print(f"     已记录 {old_fp}  →  当前 {fp}")
            print("     续跑只看 (config, arm, instance_id, rep)，不看 prompt，")
            print("     沿用同一 run_id 会跳过全部请求、产出旧模板的数据且不报错。")
            print(f"     请换 run-id 重跑，例如：--run-id {run_id}-v2")
            return 3
        if old_fp == fp:
            print(f"模板指纹    : {fp}（与已存在的 manifest 一致，续跑安全）")
        if not old_fp:
            print("\n  !! 注意：该 run 的 manifest 早于「prompt 指纹」功能，无法核验模板是否变过。")
            print("     若你刚改过模板，继续用这个 run_id 会静默复用旧模板的数据。")
            print("     保险做法：换 run-id 重跑（本 run 的产物保留备查）。")

    conc = {a.model: a.concurrency for a in selected}
    print(f"并发        : {conc}   总并发 {sum(conc.values())}")

    # ---- ★ 执行**前**先落一次 manifest -------------------------------------
    # 原先只在跑完之后写，于是：
    #   * 中断的 run **没有任何模型记录** ⇒ 续跑时无从判断"原本该跑哪些模型"，
    #     只能重读 arms.yaml —— 这正是上面那个"中途加模型污染 run"的根源；
    #   * 事后无法区分"这次跑了几模型"和"文件里恰好有几模型"。
    # 提前写一次，模型集合从第一秒起就是**确定且可查**的。
    # 跑完后会再写一次（并入已记录的模型 + 补齐 stats/耗时）。
    data_hash_pre = _hash_of(eval_path)
    execute.write_manifest(
        run_id=run_id, configs=[cfg_key.value],
        arms=[a.to_dict() for a in selected],
        data_hash=data_hash_pre,
        models_snapshot=[a.model for a in selected],
        note=f"{spec_cfg.name}｜{spec_cfg.desc}" + (f"｜limit={args.limit}" if args.limit else "")
             + "｜（执行前预写：模型集合从此确定）",
        param_overrides=overrides,
        reps=spec_cfg.reps,
        param_arms=(sorted(_tgt) if _tgt else None),
        dataset=ds_name,
    )

    # ---- 执行 --------------------------------------------------------------
    # ★ 模型级端点：只有声明了 base_url 的模型才进这个表。
    #   没有它，旗舰档会被发到聚合端点 —— 那正是它额度少、我们不用的地方。
    arm_eps = {a.name: (a.base_url, a.key_env) for a in selected if a.base_url}
    for _n, (_u, _k) in arm_eps.items():
        print(f"端点覆盖    : {_n} -> {_u}（key={_k}）")
    # ★ 模型级客户端参数：只有**显式设置**的模型才进表（None 的走 ChatClient 默认）。
    #   修的是 arms.yaml 的 `timeout_s` / `max_retries` 被 `load_arms` 弹出后丢弃、
    #   改了没有任何效果的问题。
    arm_tos = {a.name: a.timeout_s for a in selected if a.timeout_s is not None}
    arm_mrs = {a.name: a.max_retries for a in selected if a.max_retries is not None}
    if arm_tos:
        print(f"模型级超时    : {arm_tos}" + ("（被 --timeout 覆盖）" if args.timeout else ""))
    if arm_mrs:
        print(f"模型级重试    : {arm_mrs}")
    stats = execute.run_specs(
        run_id=run_id, specs=specs, arms_concurrency=conc, limit=args.limit,
        arm_endpoints=arm_eps or None,
        arm_timeouts=arm_tos or None,
        arm_retries=arm_mrs or None,
        timeout_s=args.timeout,
    )

    # ---- manifest ----------------------------------------------------------
    data_hash = _hash_of(eval_path)
    mp = execute.write_manifest(
        run_id=run_id, configs=[cfg_key.value],
        arms=[a.to_dict() for a in selected],
        data_hash=data_hash,
        models_snapshot=[a.model for a in selected],
        note=f"{spec_cfg.name}｜{spec_cfg.desc}" + (f"｜limit={args.limit}" if args.limit else ""),
        stats=stats,
        prompt_fingerprints={cfg_key.value: fp},
        param_overrides=overrides,
        reps=spec_cfg.reps,
        param_arms=(sorted(_tgt) if _tgt else None),
        dataset=ds_name,
    )
    print(f"manifest    : {mp.relative_to(paths.ROOT)}")

    # ---- 判分（双口径）----------------------------------------------------
    if args.no_tally:
        print("已跳过判分（--no-tally）")
        return 0

    results = [r for r in execute.load_results(run_id, cfg_key)]
    by_id = {i.id: i for i in instances}
    rows = []
    for r in results:
        inst = by_id.get(r.instance_id)
        if inst is None:
            continue
        # ★ C6：第 r.rep 条用的是**第 r.rep 个排列**的题面，判分必须用同一个排列，
        #   否则会把"同一个选项换了个位置"判成答错 —— 一个静默的、方向随机的错误。
        if spec_cfg.option_permutation:
            inst = build_set.permute_instance(inst, r.rep)[0]
        rows.extend(tally.tally_both_modes(r, inst))

    out_dir = paths.tallies_dir(run_id, cfg_key.value)
    tpath = out_dir / "all.jsonl"
    tpath.write_text(
        "\n".join(json.dumps(t.to_dict(), ensure_ascii=False) for t in rows) + "\n",
        encoding="utf-8",
    )

    # ---- 小结 --------------------------------------------------------------
    # ★ 准确率一律走 metrics.tier_accounting（项目唯一的四档口径）。
    #   这里曾经自己算 `correct / 全部`（HELM 式分母），把 invalid / truncated
    #   算成了答错，于是同一批数据出现两套准确率：
    #     C2 严格口径打印 0.0%（真实：无定义，410/450 判不出来）
    #     C5 严格口径打印 4.9%（真实：66.7%）
    #   两套分母共存比任何一个错值都危险，故不再本地计算。
    print("\n" + "=" * 76)
    print("判分小结")
    print("=" * 76)
    print(f"  结果 {len(results)} 条 → Tally {len(rows)} 条（双口径）")
    print(f"  {'口径':<9}{'准确率':>8}{'可判':>7}{'无效':>8}{'截断':>8}   档位分布")
    for mode in (ParseMode.STRICT, ParseMode.LENIENT):
        ac = metrics.tier_accounting([t.tier.value for t in rows if t.mode is mode])
        inv = f"{ac['invalid_rate']:.1%}" if ac["invalid_rate"] is not None else "—"
        trc = f"{ac['truncated_rate']:.1%}" if ac["truncated_rate"] is not None else "—"
        print(f"  {mode.value:<9}{metrics.format_acc(ac['acc']):>8}"
              f"{ac['n_gradeable']:>7}{inv:>8}{trc:>8}   {ac['tiers']}")

    n_trunc = sum(1 for t in rows if t.tier is Tier.TRUNCATED)
    if n_trunc:
        print(f"\n  !! {n_trunc} 条 truncated —— 检查 max_tokens 与思考开关")

    print("\n  按模型准确率（严格口径；分母 = correct + wrong，不含 invalid/truncated）：")
    for a in selected:
        ac = metrics.tier_accounting(
            [t.tier.value for t in rows if t.mode is ParseMode.STRICT and t.arm == a.model])
        print(f"    {a.name:<28}{ac['n_correct']:>4}/{ac['n_gradeable']:<4} "
              f"{metrics.format_acc(ac['acc']):>8}   {ac['tiers']}")

    print(f"\n  Tally -> {tpath.relative_to(paths.ROOT)}")
    return 0


def _coerce(v: str):
    """把 CLI 字符串转成合适类型：bool → int → float → str。"""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v


def _hash_of(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
