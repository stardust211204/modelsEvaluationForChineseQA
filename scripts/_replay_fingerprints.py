"""用当前代码重放 11 个能力批次的 prompt 指纹，与各自 manifest 比对。

指纹算法与 run_config.py 一致：
    adapt.prompt_fingerprint(instances, template, demos,
                             permutations=reps if option_permutation else 1)
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

from meval import adapt, build_set, paths

SPECS = {s.key.value: s for s in adapt.load_matrix()}
VP = {i.id: i for i in build_set.load_frozen(paths.data() / "valid_pool.jsonl")}


def prep(dsfile, cfgkey):
    spec = SPECS[cfgkey]
    ds_name = dsfile.replace(".jsonl", "")
    pool = build_set.load_frozen(paths.data() / dsfile)
    insts = pool if cfgkey == "c1" else adapt.subset_from_c1(pool, spec.n_instances)
    # ★ 必须传 dataset：c5b 的 fixed_ids_by_dataset 按数据集给不同的示例组
    demos = adapt.load_demos(spec, list(VP.values()), dataset=ds_name) if spec.few_shot else None
    tpl = (paths.ROOT / spec.prompt_template).read_text(encoding="utf-8")
    perms = spec.reps if spec.option_permutation else 1
    return adapt.prompt_fingerprint(insts, tpl, demos, permutations=perms), len(insts)


PLAN = [
    ("eval_set.jsonl", "c1", "c1-5arm-v2", 5),
    ("eval_set.jsonl", "c2", "c2hi-5arm-v2", 5),
    ("eval_set.jsonl", "c3", "c3-5arm-v2", 5),
    ("eval_set.jsonl", "c5", "c5-5arm-v3", 5),
    ("eval_set.jsonl", "c6", "c6-5arm-v2", 5),
    ("ceval_set.jsonl", "c1", "ceval-c1-5arm-v2", 5),
    ("ceval_set.jsonl", "c2", "ceval-c2hi-5arm-v2", 5),
    ("ceval_set.jsonl", "c3", "ceval-c3-5arm", 5),
    ("ceval_set.jsonl", "c5", "ceval-c5-5arm-v3", 5),
    ("ceval_set.jsonl", "c5b", "ceval-c5b-5arm", 5),
    ("ceval_set.jsonl", "c6", "ceval-c6-5arm", 5),
]

print("=" * 100)
print("  prompt 指纹重放（当前代码 vs manifest）")
print("=" * 100)
print()
print(f"  {'run_id':<24}{'题数':>6}{'模型':>4}{'reps':>6}{'调用':>8}  {'指纹':<18}{'manifest':<18} 结果")
print("  " + "-" * 92)
allok = True
for dsfile, cfg, run_id, arms in PLAN:
    fp, n = prep(dsfile, cfg)
    d = next((b / run_id for b in (paths.ROOT / "runs", paths.ROOT / "archive" / "runs")
              if (b / run_id).exists()), None)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    want = (m.get("prompt_fingerprints") or {}).get(cfg)
    reps = SPECS[cfg].reps
    calls = n * reps * arms
    ok = fp == want
    allok &= ok
    print(f"  {run_id:<24}{n:>6}{arms:>4}{reps:>6}{calls:>8}  {fp:<18}{str(want):<18} "
          f"{'OK' if ok else '差异'}")
print("  " + "-" * 92)
print(f"  全部一致：{allok}")
