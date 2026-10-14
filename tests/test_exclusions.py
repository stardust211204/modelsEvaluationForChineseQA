"""排除清单的回归测试。

这些不变量看似琐碎，但每一条被破坏都会**静默地**毁掉结论：
  * 清单规则漂移 ⇒ "我剔除了缺陷题"变成假话（剔的不是缺陷题）
  * 题集哈希被改 ⇒ 所有历史 run 判为不可配对，8,000+ 次调用作废
  * 过滤没生效 ⇒ 报告说剔了、数字没剔，是**最难发现**的一类错
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from meval import build_set, exclusions, paths  # noqa: E402


@pytest.fixture(autouse=True)
def _clear():
    exclusions.reset_cache()
    yield
    exclusions.reset_cache()


def test_list_exists_and_loads():
    """清单存在且能读出非空集合。"""
    if not exclusions.list_path().exists():
        pytest.skip("尚未生成 configs/exclusions.json")
    drop = exclusions.load()
    assert isinstance(drop, set) and drop, "清单为空 —— 要么没生成，要么解析失败"


def test_every_excluded_instance_really_hits_a_rule():
    """★ 反向检查：清单里的每一题都必须真的命中规则。

    如果这条挂了，说明清单与规则脱节 —— 那么"剔除的是缺陷题"这个说法就是假的。

    ★ 必须用 `detect_instance_defects`（文本正则 + 结构化判据）。
      只用 `detect_bad_symbols` 会把"选项重复"那几题误判成查不出缺陷。
    """
    if not exclusions.list_path().exists():
        pytest.skip("尚未生成清单")
    drop = exclusions.load()
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    missed = [k for k in drop
              if k in pool and not exclusions.detect_instance_defects(pool[k])]
    assert not missed, f"清单里有 {len(missed)} 题查不出缺陷：{missed[:5]}"


def test_duplicate_options_are_detected():
    """★ 选项重复必须能被检出。

    重复选项使"选中正确选项"失去唯一解，属机翻引入的题目缺陷。
    """
    from meval.types import Instance

    def mk(opts):
        return Instance(id="t", question="q", options=opts, gold_indices=[0])

    dup = mk(["甲", "乙", "甲", "丁"])
    assert exclusions.detect_duplicate_options(dup), "重复选项漏检"
    assert exclusions.detect_instance_defects(dup), "重复选项未进总判据"

    clean = mk(["甲", "乙", "丙", "丁"])
    assert not exclusions.detect_duplicate_options(clean), "干净选项被误判"

    # 仅空白差异不算不同项（翻译常见形态）
    ws = mk(["甲 ", "甲", "乙", "丙"])
    assert exclusions.detect_duplicate_options(ws), "去空白后重复应被检出"


def test_apply_reduces_and_removes():
    """apply 必须真的减少条目，且不留下任何被剔的题。"""
    if not exclusions.list_path().exists():
        pytest.skip("尚未生成清单")
    drop = exclusions.load()
    ids = [f"fake-{n}" for n in range(10)] + sorted(drop)[:3]
    kept = exclusions.apply(ids)
    assert len(kept) == len(ids) - min(3, len(drop) | 0 and 3) or len(kept) < len(ids)
    assert not (set(kept) & drop), "过滤后仍残留被剔除的题 —— 过滤未生效"


def test_dataset_file_hash_is_untouched():
    """★★ 最关键的守卫：排除机制**绝不允许**改动题集文件。

    题集哈希一变，`data/frozen.json` 里冻结的值就对不上，门禁会把所有历史 run
    判成"题集已变，不可配对比较"。这条测试就是那个不变量的看门人。
    """
    frozen = paths.data() / "frozen.json"
    if not frozen.exists():
        pytest.skip("尚未冻结")
    want = (json.loads(frozen.read_text(encoding="utf-8")).get("eval_set") or {}).get("sha256_16")
    if not want:
        pytest.skip("frozen.json 里没有 eval_set 哈希")
    got = hashlib.sha256((paths.data() / "eval_set.jsonl").read_bytes()).hexdigest()[:16]
    assert got == want, (
        f"题集哈希变了！实际={got} 冻结={want}\n"
        f"  排除缺陷题**必须**走 configs/exclusions.json + 分析层过滤，\n"
        f"  绝不能直接删 data/eval_set.jsonl 里的行。"
    )


def test_rule_fingerprint_matches_or_raises():
    """规则漂移必须**报错**，不能静默用旧清单。"""
    p = exclusions.list_path()
    if not p.exists():
        pytest.skip("尚未生成清单")
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("rule_fingerprint"):
        pytest.skip("清单没有规则指纹（旧版格式）")
    assert d["rule_fingerprint"] == exclusions.rule_fingerprint(), (
        "清单的规则指纹与代码不一致 —— 请重跑 scripts/build_exclusions.py"
    )


def test_detect_catches_the_four_known_patterns():
    """四类已知的 LaTeX 提取错误都要能被检出。"""
    cases = {
        r"J = 0.01e^\pir sin(\omegat)": "命令后丢空格",
        r"x^\pi y": "上标嵌套",
        r"a\alpha b": "字母后紧跟命令",
        r"值为 $ $ 厘米": "空数学环境",
    }
    for text, why in cases.items():
        assert exclusions.detect_bad_symbols(text), f"漏检：{why} —— {text!r}"
    assert not exclusions.detect_bad_symbols("普通中文题目，无公式"), "干净文本被误判"


# --------------------------------------------------------------------------
# ★★ load_tallies 默认过滤（2026-10-06 加）
#
# 改之前：过滤责任散落在每个调用方，9 个脚本读了 tallies 却忘了过滤 ——
# 同一份 tallies 在不同脚本里算出不同准确率。这两条测试锁住"默认必须过滤"。
# --------------------------------------------------------------------------

def _any_run_with_tallies():
    """找一个真有 tallies 的 run，找不到就 skip。"""
    from meval import metrics  # noqa: F401  (仅为下面的类型标注)
    runs = _ROOT / "runs"
    if not runs.exists():
        return None
    for d in sorted(runs.iterdir()):
        td = d / "tallies"
        if td.is_dir() and any(td.rglob("all.jsonl")):
            return d.name
    return None


def test_load_tallies_filters_by_default():
    """★★ `load_tallies()` 默认必须剔除排除清单里的题。

    这是防"忘记过滤"的守卫 —— 只要默认行为被改回不过滤，这条就挂。
    """
    from meval import metrics
    from meval.types import Config

    run = _any_run_with_tallies()
    if run is None:
        pytest.skip("没有可用的 run")
    if not exclusions.list_path().exists():
        pytest.skip("尚未生成清单")

    drop = exclusions.load()
    flt = metrics.load_tallies(run, Config.C1)
    raw = metrics.load_tallies(run, Config.C1, drop_excluded=False)

    assert len(flt) <= len(raw), "过滤后条目数不应增加"
    leftover = {t["instance_id"] for t in flt} & drop
    assert not leftover, (
        f"默认读取后仍残留 {len(leftover)} 条被剔除的题 —— 默认过滤失效了。\n"
        f"  `load_tallies` 的 `drop_excluded` 默认值必须是 True。"
    )
    # 只有在真的跑过被剔题时，两者才会不同
    if {t["instance_id"] for t in raw} & drop:
        assert len(flt) < len(raw), "存在被剔题却没被过滤掉"


def test_load_tallies_opt_out_returns_everything():
    """显式 `drop_excluded=False` 必须返回全量（门禁的计数校验依赖它）。"""
    from meval import metrics
    from meval.types import Config

    run = _any_run_with_tallies()
    if run is None:
        pytest.skip("没有可用的 run")

    raw = metrics.load_tallies(run, Config.C1, drop_excluded=False)
    p = _ROOT / "runs" / run / "tallies" / Config.C1.value / "all.jsonl"
    n_lines = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert len(raw) == n_lines, (
        f"drop_excluded=False 应返回全部 {n_lines} 条，实得 {len(raw)} 条。\n"
        f"  门禁的完整性计数依赖这个行为。"
    )
