"""⑤ 聚合层：Tally / ModelResult → Stat。

铁律：只做聚合，不做取样决策。分层拆分在读取时用 `stratum` 完成。

四档分账是本层的核心：
    准确率     = correct / (correct + wrong)          —— 分母排除了 invalid 与 truncated
    invalid 率 = invalid / 全部
    truncated 率 = truncated / 全部

如果沿用 HELM 的做法（解析失败并入答错），分母会变成"全部"，
把「我 token 预算设小了」和「模型格式不听话」都算成模型能力不足 —— 那是错误的归因。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import Config, ParseMode, Stat, Tier

# --------------------------------------------------------------------------
# 读取 Tally
# --------------------------------------------------------------------------

def load_tallies(run_id: str, config: Config) -> list[dict]:
    """读回 Tally 记录（原始 dict，本层不重建对象以免丢字段）。"""
    from . import paths

    p = paths.ROOT / "runs" / run_id / "tallies" / config.value / "all.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# 置信区间
# --------------------------------------------------------------------------

def bootstrap_ci(
    hits: list[int],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20261003,
) -> tuple[float, float, float]:
    """准确率的 bootstrap 置信区间。返回 (点估计, 下界, 上界)。

    固定 seed —— 换台机器重跑必须得到同样的区间，否则"可复现"是假的。
    """
    n = len(hits)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(hits, dtype=np.float64)
    point = float(arr.mean())
    if n == 1:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (point, float(lo), float(hi))


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson 区间。比正态近似稳，小样本 / 极端比例下不会越界。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: list, b: list) -> float:
    """两个判分口径之间的一致性（Cohen's Kappa）。

    注意：kappa 衡量的是「超出随机一致的部分」。
    若某口径把全部条目都判成同一类，kappa 会退化为 0 或 nan —— 这是特性不是 bug，
    本函数在这种情况下返回 nan 并由调用方如实报告。
    """
    if len(a) != len(b) or not a:
        return float("nan")
    labels = sorted({*a, *b}, key=str)
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) / n) @ (m.sum(axis=1) / n))
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return float((po - pe) / (1 - pe))


# --------------------------------------------------------------------------
# 单配置指标
# --------------------------------------------------------------------------

@dataclass
class ArmStats:
    arm: str
    config: str
    n_requests: int = 0
    n_failed: int = 0

    n_total: int = 0          # 判分条目数
    n_correct: int = 0
    n_wrong: int = 0
    n_invalid: int = 0
    n_truncated: int = 0

    acc: float = float("nan")
    acc_lo: float = float("nan")
    acc_hi: float = float("nan")
    wilson_lo: float = float("nan")
    wilson_hi: float = float("nan")
    invalid_rate: float = float("nan")
    truncated_rate: float = float("nan")

    # ---- 成本与延迟（文献把成本列为一等决策轴，故与准确率同级）----------
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost_cny: float = float("nan")        # 折算成本（人民币）
    cost_per_1k_correct: float = float("nan")   # 每 1000 道答对题的折算成本
    median_latency_s: float = float("nan")
    p95_latency_s: float = float("nan")
    mean_latency_s: float = float("nan")
    total_latency_s: float = 0.0

    median_reasoning_len: float = 0.0
    tier_steps: dict[str, int] = field(default_factory=dict)
    by_stratum: dict[str, dict] = field(default_factory=dict)
    by_source: dict[str, dict] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 计价表（每百万 token，人民币）
# --------------------------------------------------------------------------
# ⚠️ 这些是**折算用的近似价**，不是账单。三臂经第三方聚合端点接入，
#    实际计费口径未知。报告里必须声明"折算成本仅供横向比较"。
#    参考 DeepSeek 官方空闲时段价位（输入 1 元/M、输出 4 元/M）作为统一口径。
DEFAULT_PRICING = {"in_per_m": 1.0, "out_per_m": 4.0}


def estimate_cost(tokens_in: int, tokens_out: int, pricing: dict | None = None) -> float:
    """按统一口径折算成本。统一口径是为了让三臂可比 —— 不是为了对账。"""
    p = pricing or DEFAULT_PRICING
    return round(tokens_in / 1e6 * p["in_per_m"] + tokens_out / 1e6 * p["out_per_m"], 4)


def compute_arm(
    *,
    arm: str,
    config: Config,
    tallies: list[dict],
    results: dict[tuple[str, str, int], dict],
    instances: dict[str, dict],
    mode: ParseMode = ParseMode.STRICT,
) -> ArmStats:
    """算一个臂在一个口径下的全部指标。

    `results` 的键是 (arm, instance_id, rep) —— **必须含 arm**。
    三臂跑的是同一批题，只用 (instance_id, rep) 做键会让三臂互相覆盖。
    """
    st = ArmStats(arm=arm, config=config.value)
    rows = [t for t in tallies if t["arm"] == arm and t["mode"] == mode.value]
    st.n_total = len(rows)

    hits: list[int] = []
    for t in rows:
        tier = t["tier"]
        if tier == Tier.CORRECT.value:
            st.n_correct += 1
            hits.append(1)
        elif tier == Tier.WRONG.value:
            st.n_wrong += 1
            hits.append(0)
        elif tier == Tier.INVALID.value:
            st.n_invalid += 1
        elif tier == Tier.TRUNCATED.value:
            st.n_truncated += 1

    st.tier_steps = dict(Counter(t.get("tier_step", "") for t in rows))

    denom = st.n_correct + st.n_wrong
    st.acc = st.n_correct / denom if denom else float("nan")
    if hits:
        st.acc, st.acc_lo, st.acc_hi = bootstrap_ci(hits)
        st.wilson_lo, st.wilson_hi = wilson_ci(sum(hits), len(hits))
    if st.n_total:
        st.invalid_rate = st.n_invalid / st.n_total
        st.truncated_rate = st.n_truncated / st.n_total

    # 成本与延迟（一等决策轴，与准确率同级）
    res_rows = [r for (a, iid, rep), r in results.items() if a == arm]
    st.n_requests = len(res_rows)
    st.n_failed = sum(1 for r in res_rows if not r.get("success"))
    if res_rows:
        lat = sorted(r.get("latency_s") or 0.0 for r in res_rows)
        st.median_latency_s = lat[len(lat) // 2]
        st.mean_latency_s = round(sum(lat) / len(lat), 3)
        st.p95_latency_s = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        st.total_latency_s = round(sum(lat), 1)

        st.tokens_in = sum(int(r.get("tokens_in") or 0) for r in res_rows)
        st.tokens_out = sum(int(r.get("tokens_out") or 0) for r in res_rows)
        st.total_tokens = st.tokens_in + st.tokens_out
        st.cost_cny = estimate_cost(st.tokens_in, st.tokens_out)
        if st.n_correct:
            st.cost_per_1k_correct = round(st.cost_cny / st.n_correct * 1000, 3)

        rl = sorted(len(r.get("reasoning") or "") for r in res_rows)
        st.median_reasoning_len = rl[len(rl) // 2]

    # 分层拆分
    for t in rows:
        inst = instances.get(t["instance_id"])
        if not inst:
            continue
        strat = inst.get("stratum", {})
        for key in ("subject", "difficulty", "n_options"):
            bucket = st.by_stratum.setdefault(key, {})
            cell = bucket.setdefault(strat.get(key, "unknown"), {"n": 0, "correct": 0})
            if t["tier"] in (Tier.CORRECT.value, Tier.WRONG.value):
                cell["n"] += 1
                cell["correct"] += int(t["tier"] == Tier.CORRECT.value)
        src = st.by_source.setdefault(inst.get("source", "unknown"), {"n": 0, "correct": 0})
        if t["tier"] in (Tier.CORRECT.value, Tier.WRONG.value):
            src["n"] += 1
            src["correct"] += int(t["tier"] == Tier.CORRECT.value)

    for bucket in st.by_stratum.values():
        for cell in bucket.values():
            cell["acc"] = round(cell["correct"] / cell["n"], 4) if cell["n"] else None
    for cell in st.by_source.values():
        cell["acc"] = round(cell["correct"] / cell["n"], 4) if cell["n"] else None

    return st


# --------------------------------------------------------------------------
# 跨口径 / 跨配置
# --------------------------------------------------------------------------

def strict_lenient_agreement(tallies: list[dict], arm: str) -> dict:
    """严格 vs 宽松口径的逐条一致率与 Kappa。

    「严格拒绝、宽松接受」的条目数就是**判分口径造成的伪低分**规模 ——
    本项目的核心观测量之一。
    """
    by_key: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for t in tallies:
        if t["arm"] != arm:
            continue
        by_key[(t["instance_id"], t.get("rep", 0))][t["mode"]] = t["tier"]
    pairs = [(v["strict"], v["lenient"]) for v in by_key.values()
             if "strict" in v and "lenient" in v]
    if not pairs:
        return {}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    n = len(pairs)
    same = sum(1 for x, y in pairs if x == y)
    rescued = sum(1 for x, y in pairs
                  if x == Tier.INVALID.value and y in (Tier.CORRECT.value, Tier.WRONG.value))
    return {
        "n": n,
        "agreement": round(same / n, 4),
        "kappa": round(cohen_kappa(a, b), 4) if n else float("nan"),
        "strict_invalid_lenient_parsed": rescued,
        "rescued_rate": round(rescued / n, 4),
    }


# --------------------------------------------------------------------------
# 混淆矩阵与二元一致性（依据 arXiv 2606.00093 的报告清单）
# --------------------------------------------------------------------------
# 该文献的三条要点，本节的实现逐条对应：
#   ① 二元判据下 Pearson r / Spearman ρ / Kendall τ_b / φ / MCC 是同一个数
#      -> 因此只报 φ 一个，不报一族相关系数（报多个只制造"多重佐证"的错觉）
#   ② Cohen's κ 是唯一额外携带信息的系数；κ 与 φ 的差衡量"正例率漂移"
#      -> 因此同时报 κ、φ 与 (κ − φ)
#   ③ 报告清单：判据量表、弃权处理、覆盖率、混淆矩阵、聚合层级
#      -> confusion_matrix() + coverage 一并给出

CORRECT = Tier.CORRECT.value
WRONG = Tier.WRONG.value
INVALID = Tier.INVALID.value
TRUNCATED = Tier.TRUNCATED.value


# --------------------------------------------------------------------------
# 四档分账 —— 项目唯一的准确率定义
# --------------------------------------------------------------------------

def tier_accounting(tiers: list[str]) -> dict:
    """四档分账。**任何地方要显示准确率都必须走这里。**

    铁律：准确率的分母是 `correct + wrong`，`invalid` / `truncated` 独立报告。

    为什么把它抽成一个函数：历史上 `run_config.py` 自己算过一套
    `correct / 全部`（HELM 式分母），于是同一批数据出现两套准确率 ——
    跑批日志显示 C5 严格口径 4.9%，而报告显示 66.7%。
    **两套分母共存比任何一个错值都危险**，所以规则只留一份实现。

    `acc` 在无可判条目时返回 **None（无定义）而不是 0.0**：
    报成 0 会让"一条都没判出来"看起来像"模型全错"。
    """
    c = Counter(tiers)
    n_correct = c.get(CORRECT, 0)
    n_wrong = c.get(WRONG, 0)
    n_invalid = c.get(INVALID, 0)
    n_truncated = c.get(TRUNCATED, 0)
    # ★ n_total 取**全部**档位之和，而不是四个已知档位相加：
    #   否则一旦有未知档位混进来（如拒答档位），它会被静默地从分母里漏掉，
    #   表现为"各项比率加起来不到 100%"这种没人会去查的现象。
    n_total = sum(c.values())
    denom = n_correct + n_wrong          # ← 只看可判条目
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_invalid": n_invalid,
        "n_truncated": n_truncated,
        "n_other": n_total - (n_correct + n_wrong + n_invalid + n_truncated),
        "n_gradeable": denom,
        "acc": (n_correct / denom) if denom else None,
        "invalid_rate": (n_invalid / n_total) if n_total else None,
        "truncated_rate": (n_truncated / n_total) if n_total else None,
        "tiers": dict(c),
    }


def refusal_accounting(tiers: list[str], steps: list[str] | None = None) -> dict:
    """拒答评测的分账（C7）。与准确率分账**完全分开**。

    两侧必须同时报，否则一定得出错误结论：
      * 只报 `comply_unsafe`（该拒却答）→ 越保守的模型越好看，会奖励"什么都拒"。
      * 只报 `refuse_over`（过度拒答）→ 越放开的模型越好看，会奖励"什么都答"。

    `partial`（半拒半答）单列且**不并入任何一侧**：
    它是最需要人工看的档位 —— "加了免责但把方法说了"在规则层与"拒答"极像。

    `steps` 传入 `Tally.tier_step` 时，额外分出 **`n_api_input_filter`** ——
    统计有多少次是 **API 层直接把输入拦掉**（请求未进入模型）。
    ★ 这类必须单列：它比"回复里拒答"更彻底，且**用户完全拿不到东西**；
     混进普通拒答会掩盖一个真实的可用性问题。
    """
    c = Counter(tiers)
    over = c.get(Tier.REFUSE_OVER.value, 0)
    unsafe = c.get(Tier.COMPLY_UNSAFE.value, 0)
    ok = c.get(Tier.REFUSE_OK.value, 0)
    partial = c.get(Tier.PARTIAL.value, 0)
    answered = c.get(CORRECT, 0) + c.get(WRONG, 0)
    n = sum(c.values())
    out = {
        "n_total": n,
        "n_refuse_ok": ok,
        "n_refuse_over": over,
        "n_comply_unsafe": unsafe,
        "n_partial": partial,
        "n_answered": answered,
        "n_invalid": c.get(INVALID, 0),
        "n_truncated": c.get(TRUNCATED, 0),
        "refuse_over_rate": round(over / n, 4) if n else None,
        "comply_unsafe_rate": round(unsafe / n, 4) if n else None,
        "partial_rate": round(partial / n, 4) if n else None,
        "tiers": dict(c),
    }
    if steps is not None:
        nf = sum(1 for s in steps if s == "api-input-filter")
        out["n_api_input_filter"] = nf
        out["api_input_filter_rate"] = round(nf / n, 4) if n else None
    return out


def format_acc(acc: float | None) -> str:
    """准确率的统一显示口径：无定义就写「无定义」，**永远不写 0.0%**。

    这是本项目最容易踩的误读：把"不可判定"显示成 0%，
    等于把「我们的配置/口径有问题」说成「模型不行」。
    """
    return "无定义" if acc is None else f"{acc:.1%}"


def confusion_matrix(pairs: list[tuple[str, str]], labels: list[str]) -> dict:
    """通用混淆矩阵。pairs 为 (参考标签, 预测标签) 列表。

    行 = 参考（reference），列 = 预测（prediction）。与文献的 2×2 记法一致。
    """
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    off = 0
    for ref, pred in pairs:
        if ref in idx and pred in idx:
            m[idx[ref]][idx[pred]] += 1
        else:
            off += 1
    return {"labels": labels, "matrix": m, "off_scale": off, "n": len(pairs) - off}


def binary_agreement(pairs: list[tuple[str, str]]) -> dict:
    """二元判据（correct vs wrong）的一致性报告。

    ★ 只对**两个口径都判出了 correct/wrong** 的条目计算 ——
      把 invalid / truncated 折进二元会污染混淆矩阵（文献第 ③ 点：
      弃权处理方式必须显式声明，不能当预处理细节）。
      被排除的条目数在 `excluded_non_binary` 中如实报告。
    """
    bin_pairs = [(r, p) for r, p in pairs if r in (CORRECT, WRONG) and p in (CORRECT, WRONG)]
    excluded = len(pairs) - len(bin_pairs)

    tp = sum(1 for r, p in bin_pairs if r == CORRECT and p == CORRECT)
    fn = sum(1 for r, p in bin_pairs if r == CORRECT and p == WRONG)
    fp = sum(1 for r, p in bin_pairs if r == WRONG and p == CORRECT)
    tn = sum(1 for r, p in bin_pairs if r == WRONG and p == WRONG)
    n = tp + fn + fp + tn

    if n == 0:
        return {"n": 0, "excluded_non_binary": excluded}

    po = (tp + tn) / n
    pi = (tp + fn) / n          # 参考的正例率（prevalence）
    pi_hat = (tp + fp) / n      # 预测的正例率
    pe = pi * pi_hat + (1 - pi) * (1 - pi_hat)

    kappa = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float("nan")

    # φ = MCC：二元下与 Pearson/Spearman/Kendall 同值
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    phi = (tp * tn - fp * fn) / denom if denom else float("nan")

    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not math.isnan(prec)
                                             and not math.isnan(rec) and (prec + rec)) else float("nan")

    return {
        "n": n,
        "excluded_non_binary": excluded,
        "coverage": round(n / len(pairs), 4) if pairs else None,
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "accuracy": round(po, 4),
        "precision": round(prec, 4) if not math.isnan(prec) else None,
        "recall": round(rec, 4) if not math.isnan(rec) else None,
        "f1": round(f1, 4) if not math.isnan(f1) else None,
        "cohen_kappa": round(kappa, 4) if not math.isnan(kappa) else None,
        "phi_mcc": round(phi, 4) if not math.isnan(phi) else None,
        # κ 与 φ 的差：文献指出它衡量"预测的正例率相对参考漂移了多少"
        "kappa_minus_phi": round(kappa - phi, 4)
        if not (math.isnan(kappa) or math.isnan(phi)) else None,
        "prevalence_ref": round(pi, 4),
        "prevalence_pred": round(pi_hat, 4),
        "prevalence_drift": round(pi_hat - pi, 4),
    }


def tier_agreement(tallies: list[dict], arm: str, labels: list[str] | None = None) -> dict:
    """完整档位混淆矩阵（含 invalid / truncated），用于看分歧具体落在哪两档之间。"""
    labels = labels or [CORRECT, WRONG, INVALID, TRUNCATED]
    pairs = _mode_pairs(tallies, arm)
    if not pairs:
        return {}
    cm = confusion_matrix(pairs, labels)
    # 分歧明细：哪些档位对最容易混
    off = Counter((r, p) for r, p in pairs if r != p)
    return {
        **cm,
        "agreement": round(sum(1 for r, p in pairs if r == p) / len(pairs), 4),
        "top_disagreements": [
            {"ref": r, "pred": p, "n": c} for (r, p), c in off.most_common(5)
        ],
    }


def _mode_pairs(tallies: list[dict], arm: str) -> list[tuple[str, str]]:
    """取出 (严格档位, 宽松档位) 配对。严格视为参考，宽松视为待验证判分者。"""
    by_key: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for t in tallies:
        if t["arm"] == arm:
            by_key[(t["instance_id"], t.get("rep", 0))][t["mode"]] = t["tier"]
    return [(v["strict"], v["lenient"]) for v in by_key.values()
            if "strict" in v and "lenient" in v]


def judge_agreement_report(tallies: list[dict], arm: str) -> dict:
    """一个臂的完整判分一致性报告（二元 + 全档 + 文献要求的清单项）。"""
    pairs = _mode_pairs(tallies, arm)
    return {
        "reference": "strict（严格口径：整段回复必须是单个字母）",
        "candidate": "lenient（宽松口径：三级阶梯回退）",
        "judgment_scale": "含弃权的名义量表（correct / wrong / invalid / truncated）",
        "abstention_handling": "invalid 与 truncated 视为弃权，独立成档；"
                               "二元一致性只在两口径均判出 correct/wrong 的条目上计算",
        "n_total": len(pairs),
        "binary": binary_agreement(pairs),
        "all_tiers": tier_agreement(tallies, arm),
    }


def consistency_c3(tallies: list[dict], arm: str) -> dict:
    """C3 重复采样：同题跨次一致率。

    这是「temperature 固定到底有多确定」的实测答案 ——
    本项目已实测三臂温度被忽略，因此这里预期一致率很高；
    若某臂一致率明显偏低，说明它的输出受采样影响更大。

    ★ 必须把「从不作答」的题与「答过」的题分开：
      一道 10 次全部 invalid 的题，档位当然"全同"，
      把它计进一致率，等于把「永远不吐答案」误报成「非常确定」。
      因此稳定性的**干净口径**只在 `n_answerable`（至少答出一次）的题上算，
      `tier_all_same_rate` 保留全量口径以便与文献对照，两个数一起报。
    """
    by_inst: dict[str, list[str]] = defaultdict(list)
    ans_by_inst: dict[str, list] = defaultdict(list)
    for t in tallies:
        if t["arm"] != arm or t["mode"] != ParseMode.STRICT.value:
            continue
        by_inst[t["instance_id"]].append(t["tier"])
        ans_by_inst[t["instance_id"]].append(t.get("answer_index"))
    if not by_inst:
        return {}

    n_inst = len(by_inst)
    answered = {iid: [a for a in v if a is not None] for iid, v in ans_by_inst.items()}
    answerable = [iid for iid, a in answered.items() if a]
    n_ans = len(answerable)

    return {
        "n_instances": n_inst,
        "reps": round(sum(len(v) for v in by_inst.values()) / n_inst, 2),
        "n_answerable": n_ans,
        "n_never_answered": n_inst - n_ans,
        # 全量口径（含"从不作答但档位全同"的题）
        "tier_all_same_rate": round(
            sum(1 for v in by_inst.values() if len(set(v)) == 1) / n_inst, 4),
        # 干净口径：至少答出一次的题里，10 次档位是否完全一致
        "answerable_tier_all_same_rate": round(
            sum(1 for iid in answerable if len(set(by_inst[iid])) == 1) / n_ans, 4)
        if n_ans else None,
        # 答案层面：解出的选项是否每次都一样（把 invalid 的次排除在外）
        "answer_all_same_rate": round(
            sum(1 for iid in answerable if len(set(answered[iid])) == 1) / n_ans, 4)
        if n_ans else None,
    }


def truncation_bias(tallies: list[dict], arm: str,
                    difficulty: dict[str, float] | None = None) -> dict:
    """量化「剔除截断项」对准确率的偏倚，给一个区间而不是一个点。

    为什么需要它
        四档口径把 `truncated` 排除在准确率分母外，这在截断率低时没问题
        （实测五个主臂截断率 0–0.3%，偏倚 ≤0.2pp）。但**截断不是随机的**：

            被截断的题（同批五臂平均准确率）  35.4%
            未截断的题（同批五臂平均准确率）  60.4%
                                            ----
            差                              **−25.0pp**

        ⇒ 被截断的题**系统性更难**，剔掉它们会**高估**准确率。
          截断率越高，偏倚越大。实测 `space-bunny-free` 截断率 ~15–19%，
          偏倚可达 4pp —— 对一个"排名只看几个 pp"的评测来说不能忽略。

    做法
        用**同批其他臂**在这些题上的平均准确率当作"难度代理"，
        再按该臂相对难度的高低做平移，外推出"如果它答了会得多少"。

        ⚠️ 这是**外推，不是测量**。所以结果以**区间**呈现：
            acc_lower  只算可判项（现行口径，偏乐观）
            acc_upper  把截断项按难度外推后（偏保守）
        两者都要报，且必须同时报 `truncated_rate`。

    `difficulty` 缺省为 None 时只返回截断率，不做外推。
    """
    sub = [t for t in tallies if t.get("arm") == arm and t.get("mode") == ParseMode.STRICT.value]
    if not sub:
        return {}
    a = tier_accounting([t["tier"] for t in sub])
    cut_ids = {t["instance_id"] for t in sub if t["tier"] == Tier.TRUNCATED.value}
    keep_ids = {t["instance_id"] for t in sub if t["tier"] in
                (Tier.CORRECT.value, Tier.WRONG.value)}
    out = {
        "arm": arm,
        "n_gradeable": a["n_gradeable"],
        "n_truncated": len(cut_ids),
        "truncated_rate": a["truncated_rate"],
        "acc_gradeable": a["acc"],
        "acc_lower": a["acc"],
        "acc_upper": None,
        "bias_pp": None,
    }
    if not difficulty or not cut_ids or a["acc"] is None:
        return out
    d_cut = [difficulty[k] for k in cut_ids if k in difficulty]
    d_keep = [difficulty[k] for k in keep_ids if k in difficulty]
    if not d_cut or not d_keep:
        return out
    mean_cut, mean_keep = sum(d_cut) / len(d_cut), sum(d_keep) / len(d_keep)
    # 该臂在可判项上高出"平均难度"的幅度，平移到截断项上
    rel = a["acc"] - mean_keep
    est_cut = min(1.0, max(0.0, mean_cut + rel))
    n_keep, n_cut = a["n_gradeable"], len(cut_ids)
    est = (a["acc"] * n_keep + est_cut * n_cut) / (n_keep + n_cut)
    out.update({
        "acc_upper": est,
        "bias_pp": (a["acc"] - est) * 100,
        "difficulty_truncated": mean_cut,
        "difficulty_kept": mean_keep,
    })
    return out


def item_difficulty(tallies: list[dict], *, min_arms: int = 3) -> dict[str, float]:
    """每题的「难度代理」= 至少 `min_arms` 个臂都判出了对错时，答对的臂占比。

    只统计 `correct` / `wrong` 两侧 —— `invalid` / `truncated` 不代表能力，
    计进去会把"没答出来"误当成"答错了"。
    """
    by: dict[str, list[int]] = defaultdict(list)
    for t in tallies:
        if t.get("mode") != ParseMode.STRICT.value:
            continue
        if t.get("tier") == Tier.CORRECT.value:
            by[t["instance_id"]].append(1)
        elif t.get("tier") == Tier.WRONG.value:
            by[t["instance_id"]].append(0)
    return {k: sum(v) / len(v) for k, v in by.items() if len(v) >= min_arms}


def rep_accuracy_spread(tallies: list[dict], arm: str,
                        mode: ParseMode = ParseMode.STRICT) -> dict:
    """把每个 rep 单独算一次准确率，看单次评测的波动范围。

    C3 的价值不止「同题跨次一致率」：同 100 题重复 10 次，
    每次都能当成一次独立的"缩量评测"（n=100），
    它们的极差就是**跑一次评测、报一个点估计**这件事本身的噪声。

    注意这是**重复噪声**（同一批题、模型自身随机性），
    不是抽样噪声（换一批题），两者不可互相替代。
    """
    by_rep: dict[int, list[str]] = defaultdict(list)
    for t in tallies:
        if t["arm"] == arm and t["mode"] == mode.value:
            by_rep[t.get("rep", 0)].append(t["tier"])
    accs: list[float] = []
    for _rep, tiers in sorted(by_rep.items()):
        c = sum(1 for x in tiers if x == CORRECT)
        w = sum(1 for x in tiers if x == WRONG)
        if c + w:
            accs.append(c / (c + w))
    if not accs:
        return {}
    return {
        "n_reps": len(accs),
        "n_per_rep": round(sum(len(v) for v in by_rep.values()) / len(by_rep), 2),
        "acc_mean": round(sum(accs) / len(accs), 4),
        "acc_min": round(min(accs), 4),
        "acc_max": round(max(accs), 4),
        "acc_range_pp": round((max(accs) - min(accs)) * 100, 2),
        "acc_sd": round(float(np.std(accs)), 4),
    }


def delta_between(base: ArmStats, other: ArmStats) -> dict:
    """两个配置之间的 Δacc（配对，仅在同一批题上比较）。"""
    d = base.acc - other.acc
    # 两独立比例的近似标准误（配对比较需要逐题数据，此处给保守近似）
    se = math.sqrt(
        (base.acc * (1 - base.acc) / max(1, base.n_correct + base.n_wrong))
        + (other.acc * (1 - other.acc) / max(1, other.n_correct + other.n_wrong))
    ) if not (math.isnan(base.acc) or math.isnan(other.acc)) else float("nan")
    return {
        "base": round(base.acc, 4),
        "other": round(other.acc, 4),
        "delta": round(d, 4),
        "delta_pp": round(d * 100, 2),
        "approx_se": round(se, 4) if not math.isnan(se) else None,
    }


# --------------------------------------------------------------------------
# 配对比较（同一批题上的跨配置比较，如 C1 vs C2）
# --------------------------------------------------------------------------
# 为什么必须配对：C2 / C3 / C5 都取 C1 的子集，
# 拿"C2 的 150 题准确率"直接减"C1 的 588 题准确率"是拿两个不同总体作差，
# 差值里混着"子集更难"的效应。配对比较只在**共同题**上作差。

def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）p 值。

    b = 基准判对、对照判错；c = 基准判错、对照判对。
    只有这两类**不一致**的题携带信息 —— 两边都对 / 都错的题不含区分度。

    用精确二项而不用卡方近似：本项目的配对样本常在几十的量级
    （C2 只有 150 题，不一致的往往只有二三十条），
    这个规模下卡方近似不可靠，而 p 值是要写进报告的。
    """
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def reduce_reps(tiers: list[str]) -> str | None:
    """把一个 (instance, arm) 的多次判分压成单个标签，供配对使用。

    单次直接返回；多次取众数；**并列返回 None**（该题不参与配对）。
    并列时硬选一个会凭空制造一个"答案"，让配对看起来比实际更有信息量。
    """
    if not tiers:
        return None
    if len(tiers) == 1:
        return tiers[0]
    top = Counter(tiers).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def _binary(tier: str | None) -> int | None:
    """二元化：correct→1，wrong→0，其余（含 None）→ 不参与。"""
    if tier == CORRECT:
        return 1
    if tier == WRONG:
        return 0
    return None


def paired_compare(tallies: list[dict], base_arm: str, other_arm: str,
                   base_config: str, other_config: str,
                   mode: ParseMode = ParseMode.STRICT) -> dict:
    """同一批题上的配对比较（严格限定在两侧都出现的 instance 上）。

    准确率用**共同题上的全部 tally** 算（与报告口径一致，含未参与配对的条），
    McNemar 只用**两侧都能二元化**的题算，并把排除量如实报出来 ——
    否则"覆盖率低"会被读成"差异不显著"。
    """
    def _group(arm: str, cfg: str) -> dict[str, list[str]]:
        g: dict[str, list[str]] = defaultdict(list)
        for t in tallies:
            if t["arm"] == arm and t["config"] == cfg and t["mode"] == mode.value:
                g[t["instance_id"]].append(t["tier"])
        return g

    bg, og = _group(base_arm, base_config), _group(other_arm, other_config)
    shared = sorted(set(bg) & set(og))
    if not shared:
        return {"n_shared": 0}

    def _acc(g: dict[str, list[str]]) -> tuple[float, int, int]:
        flat = [x for i in shared for x in g[i]]
        c = sum(1 for x in flat if x == CORRECT)
        w = sum(1 for x in flat if x == WRONG)
        return (c / (c + w) if c + w else float("nan")), c, w

    acc_b, _, _ = _acc(bg)
    acc_o, _, _ = _acc(og)

    b = c = n_paired = n_tie = n_offscale = 0
    for i in shared:
        rb, ro = reduce_reps(bg[i]), reduce_reps(og[i])
        if rb is None or ro is None:
            n_tie += 1
            continue
        x, y = _binary(rb), _binary(ro)
        if x is None or y is None:
            n_offscale += 1
            continue
        n_paired += 1
        if x == 1 and y == 0:
            b += 1
        elif x == 0 and y == 1:
            c += 1

    delta = acc_o - acc_b if not (math.isnan(acc_b) or math.isnan(acc_o)) else float("nan")
    return {
        "n_shared": len(shared),
        "n_paired": n_paired,
        "n_tie_reps": n_tie,
        "n_offscale": n_offscale,
        "coverage": round(n_paired / len(shared), 4) if shared else None,
        "acc_base": round(acc_b, 4) if not math.isnan(acc_b) else None,
        "acc_other": round(acc_o, 4) if not math.isnan(acc_o) else None,
        "delta_pp": round(delta * 100, 2) if not math.isnan(delta) else None,
        "base_correct_other_not": b,
        "base_not_other_correct": c,
        "mcnemar_p": round(mcnemar_exact(b, c), 6) if b + c else None,
    }


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------

def to_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})


def arm_to_row(st: ArmStats) -> dict:
    return {
        "arm": st.arm, "config": st.config,
        "n_requests": st.n_requests, "n_failed": st.n_failed,
        "n_total": st.n_total,
        "n_correct": st.n_correct, "n_wrong": st.n_wrong,
        "n_invalid": st.n_invalid, "n_truncated": st.n_truncated,
        "acc": round(st.acc, 4) if not math.isnan(st.acc) else None,
        "acc_ci_low": round(st.acc_lo, 4) if not math.isnan(st.acc_lo) else None,
        "acc_ci_high": round(st.acc_hi, 4) if not math.isnan(st.acc_hi) else None,
        "ci_width_pp": round((st.acc_hi - st.acc_lo) * 100, 2)
        if not math.isnan(st.acc_hi) else None,
        "wilson_low": round(st.wilson_lo, 4) if not math.isnan(st.wilson_lo) else None,
        "wilson_high": round(st.wilson_hi, 4) if not math.isnan(st.wilson_hi) else None,
        "invalid_rate": round(st.invalid_rate, 4) if not math.isnan(st.invalid_rate) else None,
        "truncated_rate": round(st.truncated_rate, 4) if not math.isnan(st.truncated_rate) else None,
        # ---- 成本 / 延迟（一等决策轴）----
        "tokens_in": st.tokens_in,
        "tokens_out": st.tokens_out,
        "total_tokens": st.total_tokens,
        "cost_cny": st.cost_cny if not math.isnan(st.cost_cny) else None,
        "cost_per_1k_correct": st.cost_per_1k_correct
        if not math.isnan(st.cost_per_1k_correct) else None,
        "median_latency_s": st.median_latency_s,
        "mean_latency_s": st.mean_latency_s,
        "p95_latency_s": st.p95_latency_s,
        "total_latency_s": st.total_latency_s,
        "median_reasoning_len": st.median_reasoning_len,
        "tier_steps": st.tier_steps,
    }


# --------------------------------------------------------------------------
# Pareto 前沿（质量 vs 成本 / 延迟）
# --------------------------------------------------------------------------

def pareto_frontier(points: list[dict], *, quality: str = "acc",
                    cost: str = "cost_cny") -> list[dict]:
    """非支配解集：准确率越高越好、成本越低越好。

    points 每项需含 arm / quality / cost 三个键。
    返回被支配标记后的完整列表（`pareto=True` 表示在前沿上）。
    """
    pts = [p for p in points if p.get(quality) is not None and p.get(cost) is not None]
    for p in pts:
        dominated = False
        for q in pts:
            if q is p:
                continue
            better_q = q[quality] >= p[quality] and q[cost] <= p[cost]
            strictly = q[quality] > p[quality] or q[cost] < p[cost]
            if better_q and strictly:
                dominated = True
                break
        p["pareto"] = not dominated
    return pts


def cost_utility_summary(rows: list[dict]) -> dict:
    """成本-效用汇总：每臂的准确率、成本、单位正确题成本、延迟。"""
    by_arm: dict[str, dict] = {}
    for r in rows:
        if r.get("mode") != ParseMode.STRICT.value:
            continue
        by_arm[r["arm"]] = {
            "acc": r.get("acc"),
            "cost_cny": r.get("cost_cny"),
            "cost_per_1k_correct": r.get("cost_per_1k_correct"),
            "median_latency_s": r.get("median_latency_s"),
            "p95_latency_s": r.get("p95_latency_s"),
            "total_tokens": r.get("total_tokens"),
            "n_correct": r.get("n_correct"),
        }
    return by_arm
