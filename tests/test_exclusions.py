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
    """
    if not exclusions.list_path().exists():
        pytest.skip("尚未生成清单")
    drop = exclusions.load()
    pool = {i.id: i for i in build_set.load_frozen(paths.data() / "eval_set.jsonl")}
    missed = [k for k in drop
              if k in pool and not exclusions.detect_bad_symbols(
                  pool[k].question + " " + " ".join(pool[k].options))]
    assert not missed, f"清单里有 {len(missed)} 题查不出坏符号：{missed[:5]}"


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
