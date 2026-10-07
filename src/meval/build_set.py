"""① 造题层：分层抽样。

铁律：产出后**冻结并哈希**，后续任何模块不得再改题集。
否则 C1 / C2 / C5 的配对比较会失效（它们必须建立在同一批题上）。

分层三个维度：
    学科      按 MMLU-ProX 的学科标签配额抽样
    难度代理  用一个小模型的通过率分档（易 ≥70% / 中 40–70% / 难 <40%）
    选项数    ≤4 / 5–7 / ≥8 三档 —— 选项数直接决定随机基线，是准确率的混淆变量
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import paths
from .types import Instance, Source, Split

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

N_EVAL = 600           # 评测集总题数
SEED = 20261003        # 固定种子：抽样必须可复现

DIFFICULTY_BINS = (
    ("easy", 0.70, 1.01),   # 基线通过率 >= 0.70
    ("mid", 0.40, 0.70),
    ("hard", -0.01, 0.40),
)


def _opt_bin(n: int) -> str:
    if n <= 4:
        return "opt_le4"
    if n <= 7:
        return "opt_5_7"
    return "opt_ge8"


def _difficulty_bin(rate: float | None) -> str:
    if rate is None:
        return "unknown"
    for name, lo, hi in DIFFICULTY_BINS:
        if lo <= rate < hi:
            return name
    return "unknown"


# --------------------------------------------------------------------------
# 读公开集
# --------------------------------------------------------------------------

_HF_SPLIT_TO_ENUM = {
    "train": Split.TRAIN,
    "validation": Split.VALID,
    "valid": Split.VALID,
    "test": Split.TEST,
}


def load_public(lang: str = "zh", split: str = "test") -> list[Instance]:
    """读 MMLU-ProX-Lite 指定语言切片。

    ★ `Instance.split` 必须跟着 HF 的 `split` 走，不能写死 TEST。
      写死的后果很隐蔽：拿 validation 切片建出来的示例池，
      每一条的 split 都标成 test，于是 `adapt.load_demos()` 把它们全部过滤掉，
      最后表现为"few-shot 示例池为空"——看起来像数据问题，其实是标签写错。

    ★ valid 切片的 id 带 `valid-` 段：HF 的 question_id 在 test / validation
      两个切片里各自从 0 开始，不加区分就会撞号（撞号会让配对比较静默串题）。
      test 切片保持原有 id 方案不变，以免改动已冻结的题集哈希。
    """
    import os  # noqa: PLC0415

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset  # noqa: PLC0415

    if split not in _HF_SPLIT_TO_ENUM:
        raise ValueError(f"未知 split：{split}（可选 {sorted(_HF_SPLIT_TO_ENUM)}）")
    split_enum = _HF_SPLIT_TO_ENUM[split]
    tag = "" if split_enum is Split.TEST else f"{split_enum.value}-"

    ds = load_dataset("li-lab/MMLU-ProX-Lite", lang, split=split)
    out: list[Instance] = []
    for row in ds:
        options = [row.get(f"option_{i}") for i in range(10)]
        options = [o for o in options if o]
        out.append(
            Instance(
                id=f"prox-{lang}-{tag}{row['question_id']}",
                question=row["question"],
                options=options,
                gold_indices=[int(row["answer_index"])],
                split=split_enum,
                stratum={"subject": str(row.get("category", "unknown")), "n_options": _opt_bin(len(options))},
                source=Source.PUBLIC,
            )
        )
    return out


# --------------------------------------------------------------------------
# 分层抽样
# --------------------------------------------------------------------------

def stratify(
    pool: list[Instance],
    difficulty: dict[str, float] | None = None,
) -> list[Instance]:
    """把难度代理打进 stratum。difficulty 为 {instance_id: 通过率}。"""
    difficulty = difficulty or {}
    out = []
    for inst in pool:
        s = dict(inst.stratum)
        s["difficulty"] = _difficulty_bin(difficulty.get(inst.id))
        out.append(
            Instance(
                id=inst.id, question=inst.question, options=inst.options,
                gold_indices=inst.gold_indices, split=inst.split,
                stratum=s, source=inst.source, note=inst.note,
            )
        )
    return out


def sample_stratified(
    pool: list[Instance],
    n: int,
    *,
    seed: int = SEED,
    keys: tuple[str, ...] = ("subject", "difficulty"),
) -> list[Instance]:
    """按 keys 分层配额抽样。

    配额算法：每个 stratum 至少 1 题，余量按该层在全量中的占比分配。
    小层不会被饿死 —— 这是分层抽样与"随机抽 600 题"的关键差别。

    关于 keys 的选择（实测依据）：
      MMLU-ProX-Lite zh 切片的选项数分布极度不均 —— 548/588 是 10 选项
      （opt_ge8），只有 27 条 ≤4 选项。把 n_options 当分层键几乎不产生区分度，
      反而会把样本切碎。因此 **n_options 只作为 stratum 属性保留用于分析**，
      不参与分层。这是"先看数据再定分层键"的一个具体例子。
    """
    rng = random.Random(seed)
    buckets: dict[tuple, list[Instance]] = defaultdict(list)
    for inst in pool:
        buckets[tuple(inst.stratum.get(k, "unknown") for k in keys)].append(inst)

    if len(pool) <= n:
        return list(pool)

    # 配额：先保底 1，再按占比分配
    total = len(pool)
    quota: dict[tuple, int] = {}
    for k, items in buckets.items():
        quota[k] = max(1, round(n * len(items) / total))

    # 修正到恰好 n：多的削、少的补（削/补都优先动最大的层）
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    cur = sum(quota.values())
    i = 0
    while cur > n and i < len(order) * 3:
        k = order[i % len(order)]
        if quota[k] > 1:
            quota[k] -= 1
            cur -= 1
        i += 1
    i = 0
    while cur < n and i < len(order) * 3:
        k = order[i % len(order)]
        if quota[k] < len(buckets[k]):
            quota[k] += 1
            cur += 1
        i += 1

    picked: list[Instance] = []
    for k, items in buckets.items():
        take = min(quota[k], len(items))
        picked.extend(rng.sample(items, take))
    rng.shuffle(picked)
    return picked[:n]


# --------------------------------------------------------------------------
# 选项乱序（C6：位置敏感性）
# --------------------------------------------------------------------------

def permute_instance(inst: Instance, k: int, *, base_seed: int = SEED) -> tuple[Instance, list[int]]:
    """确定性地重排选项，返回 (新 Instance, 排列 perm)。

    `perm` 的语义：**新 Instance 的第 i 个选项 = 原 Instance 的第 perm[i] 个选项**。
    `gold_indices` 同步重映射，所以 `tally` 的判分逻辑**一行都不用改**。

    ★ k = 0 恒为**原顺序**（不洗牌）。这样 C6 的第 0 个排列就是 C1 的题面，
      可以直接当作"同一 run 内的基线"来核验管线；真正的乱序从 k = 1 开始。

    ★ 种子由 `(instance_id, k)` 派生 —— 同一个 k 永远得到同一个排列。
      这是配对比较成立的前提，也是"可复现"的全部含义。
      （Python 的 `random.Random(str)` 用字符串的 sha512 播种，跨平台稳定。）
    """
    n = len(inst.options)
    if k == 0:
        return inst, list(range(n))

    rng = random.Random(f"{base_seed}:{inst.id}:{k}")
    perm = list(range(n))
    rng.shuffle(perm)

    new_options = [inst.options[i] for i in perm]
    # 原 gold 下标 g -> 在新顺序里的位置
    new_gold = sorted(perm.index(g) for g in inst.gold_indices)
    out = Instance(
        id=inst.id, question=inst.question, options=new_options,
        gold_indices=new_gold, split=inst.split,
        stratum=dict(inst.stratum), source=inst.source, note=inst.note,
    )
    return out, perm


def unpermute_answer(answer_idx: int | None, perm: list[int]) -> int | None:
    """把「乱序后的作答下标」还原成「原题选项下标」。

    位置敏感性分析必须比较**原题下标**：否则模型每次都选 "B" 会被误读成
    "答案很稳定"，而 B 在不同排列里是**不同的选项**。
    """
    if answer_idx is None:
        return None
    return perm[answer_idx] if 0 <= answer_idx < len(perm) else None


# --------------------------------------------------------------------------
# 冻结与哈希
# --------------------------------------------------------------------------

def freeze(instances: list[Instance], path: Path) -> str:
    """写盘并返回内容哈希。哈希进 RunManifest，保证"这批数字出自哪版题集"。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(i.to_dict(), ensure_ascii=False, sort_keys=True) for i in instances
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_frozen(path: Path) -> list[Instance]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["split"] = Split(d["split"])
        d["source"] = Source(d["source"])
        out.append(Instance(**d))
    return out


# --------------------------------------------------------------------------
# 可比性检验（借 arXiv:2410.09247 的第 ② 步）
# --------------------------------------------------------------------------

def comparability_report(a: list[Instance], b: list[Instance]) -> dict:
    """比较两个题集在「选项数」与「学科」上的分布是否接近。

    目的：证明**抽出的样本**与**完整池子**分布一致（抽样未失真）——
    自建集刻意在若干维度上偏离（那是它的用途），所以这里报告的是
    "在哪些维度上偏离、偏多少"，而不是宣称它们同分布。
    """
    def dist(xs: list[Instance], key: str) -> Counter:
        return Counter(i.stratum.get(key, "unknown") for i in xs)

    def shares(c: Counter) -> dict[str, float]:
        t = sum(c.values()) or 1
        return {k: round(v / t, 3) for k, v in c.items()}

    rep = {}
    for key in ("n_options", "subject"):
        da, db = shares(dist(a, key)), shares(dist(b, key))
        keys = sorted(set(da) | set(db))
        rep[key] = {
            "pool": {k: da.get(k, 0.0) for k in keys},
            "sample": {k: db.get(k, 0.0) for k in keys},
            "max_abs_gap": round(max(abs(da.get(k, 0) - db.get(k, 0)) for k in keys), 3),
        }
    return rep


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DIFFICULTY_FILE = "difficulty_proxy.json"


def main() -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="造题：分层抽样，产出并冻结")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--n", type=int, default=N_EVAL, help=f"评测集题数（默认 {N_EVAL}）")
    ap.add_argument("--domains-per-cell", type=int, default=0,
                    help=">0 时改用「每格固定条数」抽样，n 参数被忽略")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--valid-only", action="store_true",
                    help="只构建并冻结 few-shot 示例池 data/valid_pool.jsonl（不碰 eval_set）")
    args = ap.parse_args()

    print("=" * 76)
    print("① 造题")
    print("=" * 76)

    # ---- few-shot 示例池（独立出口，刻意不重算 eval_set）--------------------
    # ★ 不重算 eval_set 是硬要求：题集已冻结并哈希，C1/C2/C3 都建立在它之上。
    #   重算一次哪怕只差一个字节，跨配置的配对比较就全部失效（门禁会拦）。
    if args.valid_only:
        vpool = load_public(args.lang, "validation")
        subj = Counter(i.stratum.get("subject") for i in vpool)
        h = freeze(vpool, paths.data() / "valid_pool.jsonl")
        print(f"  valid 切片      : {len(vpool)} 题（lang={args.lang}，split=validation）")
        print(f"    学科数        : {len(subj)}")
        print(f"    选项数分布    : {dict(Counter(i.stratum.get('n_options') for i in vpool))}")
        print(f"\n  已冻结          : data/valid_pool.jsonl  ({len(vpool)} 题)  hash={h}")
        print("  下一步：从该池选定 5 条固定示例，回填 configs/matrix.yaml 的")
        print("          c5.few_shot.fixed_ids（回填后不再变动）")
        return 0

    pool = load_public(args.lang, "test")
    print(f"  公开集          : {len(pool)} 题 (lang={args.lang})")

    # 难度代理（可选，来自 probe 阶段产出的文件）
    diff_path = paths.data() / DIFFICULTY_FILE
    difficulty: dict[str, float] = {}
    if diff_path.exists():
        difficulty = json.loads(diff_path.read_text(encoding="utf-8"))
        print(f"  难度代理        : 已载入 {len(difficulty)} 条")
    else:
        print(f"  难度代理        : 未找到 {DIFFICULTY_FILE}，difficulty 一律标 unknown")
        print("                    （可选：用本地小模型跑一遍 588 题生成该文件）")

    pool = stratify(pool, difficulty)

    n = args.n
    if args.domains_per_cell > 0:
        n = min(len(pool), _cells(pool) * args.domains_per_cell + 1)
        print(f"  抽样方式        : 每格 {args.domains_per_cell} 条 -> 目标 {n} 题")
    picked = sample_stratified(pool, n, seed=args.seed)
    print(f"  分层抽样        : {len(picked)} 题")

    dist = Counter(tuple(sorted(i.stratum.items())) for i in picked)
    subj = Counter(i.stratum.get("subject") for i in picked)
    dif = Counter(i.stratum.get("difficulty") for i in picked)
    opt = Counter(i.stratum.get("n_options") for i in picked)
    print(f"    学科分布      : {len(subj)} 个学科")
    print(f"    难度分布      : {dict(dif)}")
    print(f"    选项数分布    : {dict(opt)}")

    # ---- 抽样保真度自检 ----------------------------------------------------
    # ★ 校验「抽出来的样本」与「完整池子」的分布在选项数/学科上是否失真。
    #   这是配对比较成立的前提：如果抽样把某些学科抽没了，C1–C7 的跨配置对比
    #   就不再代表同一批题的同一分布。
    rep = comparability_report(pool, picked)
    print(f"  抽样保真度      : 选项数 max_gap={rep['n_options']['max_abs_gap']}  "
          f"学科 max_gap={rep['subject']['max_abs_gap']}")
    if max(rep["n_options"]["max_abs_gap"], rep["subject"]["max_abs_gap"]) > 0.15:
        print("    ⚠️ 分布偏差 >15pp，分层抽样可能失真，请检查 --n / --domains-per-cell")

    h_eval = freeze(picked, paths.data() / "eval_set.jsonl")
    print(f"\n  已冻结          : data/eval_set.jsonl  ({len(picked)} 题)  hash={h_eval}")

    print("\n  下一步：把 hash 记入 manifest，然后跑 adapt + execute")
    return 0


def _cells(pool: list[Instance], keys=("subject", "difficulty")) -> int:
    return len({tuple(i.stratum.get(k, "unknown") for k in keys) for i in pool})


if __name__ == "__main__":
    raise SystemExit(main())
