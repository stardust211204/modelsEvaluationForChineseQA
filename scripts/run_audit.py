"""清点 runs/ 下所有批次：哪些支撑结论、哪些是重跑、哪些出错、哪些可归档。

为什么要脚本
    61 个目录、7 个维度（臂数、数据集、配置、记录数、时间、错误、被引用情况），
    靠人眼列一定会漏。而且「能不能归档」必须从**代码与报告的引用**反推，
    不能凭印象 —— 归档错了，图就重画不出来。

分类
    A 结论批次   直接产出报告里的数字
    B 支撑批次   支撑报告里的单项声明（Kappa、端点一致性、限额模型），但不是结论来源
    C 质量核查   仅用于项目内部质检，不进对外报告
    D 被取代     早期批次，已被 A 里的 v2/v3 取代
    E 冒烟       管线连通性测试
    F 出错/废弃  跑坏或设计作废

用法
    .venv\\Scripts\\python.exe scripts\\run_audit.py              # 只报告
    .venv\\Scripts\\python.exe scripts\\run_audit.py --write      # 写 reports/RUN_AUDIT.md
    .venv\\Scripts\\python.exe scripts\\run_audit.py --archive    # 把 C/D/E/F 移到 archive/runs/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ARCHIVE = ROOT / "archive" / "runs"

# ── A 结论批次：报告里的每个数字都出自这里 ────────────────────────────────
A_AUTHORITATIVE = {
    "c1-5arm-v2": "基线 / MMLU-ProX 588 题",
    "ceval-c1-5arm-v2": "基线 / C-Eval 583 题",
    "c2hi-5arm-v2": "自由作答 / MMLU-ProX 150 题",
    "ceval-c2hi-5arm-v2": "自由作答 / C-Eval 150 题",
    "c3-5arm-v2": "稳定性 / MMLU-ProX 100×10",
    "ceval-c3-5arm": "稳定性 / C-Eval 100×10",
    "c6-5arm-v2": "选项乱序 / MMLU-ProX 100×4",
    "ceval-c6-5arm": "选项乱序 / C-Eval 100×4",
    "c5-5arm-v3": "5-shot / MMLU-ProX 150 题",
    "ceval-c5-5arm-v3": "5-shot / C-Eval 150 题（用于论证不采用）",
    "ceval-c5b-5arm": "5-shot 对照 / C-Eval 150 题（选项数匹配）",
    "orbench-full": "安全 / 源响应 3 臂",
    "orbench-qwen": "安全 / 源响应 qwen",
    "orbench-judge-full": "安全 / judge 判定 3 臂",
    "orbench-judge-qwen": "安全 / judge 判定 qwen",
}

# ── B 支撑批次：支撑报告里的单项声明，不产结论 ────────────────────────────
B_SUPPORTING = {
    "c1-full": "c4 judge 复核的被判对象（早期 3 臂基线）",
    "c4-full2": "规则 vs judge 的 Kappa（报告 §2.1 的 0.99–1.00）",
    "c4-full": "同上，前一版",
    "endpoint-cmp": "端点一致性核查（报告 §5.3「旗舰档走独立端点」）",
    "sb-c1-eval": "限额模型 / MMLU-ProX（报告 §4）",
    "sb-c1-ceval": "限额模型 / C-Eval（报告 §4）",
}

# ── C 质量核查：项目内部用，不进对外报告 ──────────────────────────────────
C_QC = {"badcase-judge", "badcase-judge2", "badcase-judge-pilot", "gold-audit"}

# ── E 冒烟 / 探针：管线连通性测试 ─────────────────────────────────────────
E_SMOKE = {"pipe-smoke", "v4pro-smoke", "smoke-fp2", "ceval-smoke",
           "c2-qwen-smoke", "sb-smoke"}

# ── F 出错 / 设计作废 ────────────────────────────────────────────────────
F_DEAD = {
    "c7-full": "C7 早期试跑，无 manifest，被 orbench-* 取代",
    "c7-full2": "同上",
    "c7-full3": "自建中文拒答题（300 条），两侧全饱和，改投 OR-Bench",
    "c7-judge": "自建题集的 judge 判定，随题集一起作废",
    "c7-judge2": "同上",
    "orbench-pilot": "OR-Bench 100 条试水，正式批次后作废",
    "orbench-judge": "试水的 judge 判定，作废",
    "sb-c2hi-eval": "限额模型的自由作答（MMLU-ProX），报告未采用",
    "sb-c2hi-ceval": "限额模型的自由作答（C-Eval），报告未采用",
    "sb-c3-eval": "限额模型的稳定性，无 manifest、记录不完整（184/1000）",
}


def read_run(name: str) -> dict:
    p = RUNS / name
    info: dict = {"name": name, "path": p}
    mf = p / "manifest.json"
    if mf.exists():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            info["arms"] = m.get("arms", [])
            info["dataset"] = m.get("dataset", "?")
            info["configs"] = m.get("configs", [])
            info["note"] = m.get("note", "")
            info["finished"] = (m.get("finished_at") or "")[:16]
        except Exception as e:                       # noqa: BLE001
            info["manifest_error"] = str(e)
    n_raw = n_fail = 0
    arms_seen: Counter = Counter()
    errs: Counter = Counter()
    for f in p.rglob("*.jsonl"):
        if "\\raw\\" not in str(f) and "/raw/" not in str(f).replace("\\", "/"):
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            n_raw += 1
            try:
                r = json.loads(line)
            except Exception:                        # noqa: BLE001
                continue
            arms_seen[r.get("arm", "?")] += 1
            if not r.get("success"):
                n_fail += 1
                errs[r.get("error_kind") or "unknown"] += 1
    info["n_raw"] = n_raw
    info["n_fail"] = n_fail
    info["arms_seen"] = dict(arms_seen)
    info["errors"] = dict(errs)
    info["mtime"] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")
    return info


def classify(name: str) -> tuple[str, str]:
    if name in A_AUTHORITATIVE:
        return "A", A_AUTHORITATIVE[name]
    if name in B_SUPPORTING:
        return "B", B_SUPPORTING[name]
    if name in C_QC:
        return "C", "项目内部质检"
    if name in E_SMOKE:
        return "E", "管线连通性测试"
    if name in F_DEAD:
        return "F", F_DEAD[name]
    return "D", "早期批次，已被 v2/v3 取代"


def main() -> int:
    ap = argparse.ArgumentParser(description="runs/ 清点与归档")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--archive", action="store_true")
    args = ap.parse_args()

    infos = [read_run(p.name) for p in sorted(RUNS.iterdir()) if p.is_dir()]
    for i in infos:
        i["cls"], i["why"] = classify(i["name"])

    by_cls: dict[str, list[dict]] = {}
    for i in infos:
        by_cls.setdefault(i["cls"], []).append(i)

    total_raw = sum(i["n_raw"] for i in infos)
    total_fail = sum(i["n_fail"] for i in infos)

    print("=" * 100)
    print("  runs/ 清点")
    print("=" * 100)
    print(f"  批次 {len(infos)} 个｜模型调用 {total_raw:,} 次｜失败 {total_fail} 次"
          f"（{total_fail / total_raw:.2%}）")
    for k in "ABCDEF":
        v = by_cls.get(k, [])
        print(f"    {k} 类 {len(v):>2} 个｜调用 {sum(x['n_raw'] for x in v):>7,} 次")

    # 出错批次
    bad = [i for i in infos if i["n_fail"]]
    print(f"\n  【有失败请求的批次】{len(bad)} 个")
    for i in bad:
        print(f"    {i['name']:<22} {i['n_fail']:>4} / {i['n_raw']:<6} "
              f"{i['n_fail'] / i['n_raw']:>6.1%}  {i['errors']}")

    if args.archive:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        moved = []
        for k in "CDEF":
            for i in by_cls.get(k, []):
                dst = ARCHIVE / i["name"]
                if dst.exists():
                    print(f"    [跳过] {i['name']} 归档目录已存在")
                    continue
                shutil.move(str(i["path"]), str(dst))
                moved.append(i["name"])
        print(f"\n  已归档 {len(moved)} 个批次 → {ARCHIVE.relative_to(ROOT)}")
        (ARCHIVE / "_MANIFEST.txt").write_text(
            "从 runs/ 归档的批次（不支撑报告结论）\n"
            f"归档时间：{datetime.now():%Y-%m-%d %H:%M}\n\n"
            + "\n".join(f"[{classify(n)[0]}] {n:<24} {classify(n)[1]}"
                        for n in sorted(moved)) + "\n",
            encoding="utf-8")
        return 0

    if args.write:
        out = ROOT / "reports" / "RUN_AUDIT.md"
        L: list[str] = ["# runs/ 清点与归档", "",
                        f"> 生成于 {datetime.now():%Y-%m-%d %H:%M}｜"
                        f"批次 **{len(infos)}** 个｜"
                        f"模型调用 **{total_raw:,}** 次", ""]
        titles = {"A": "A 类 · 结论批次（报告数字的来源）",
                  "B": "B 类 · 支撑批次（支撑单项声明）",
                  "C": "C 类 · 质量核查（项目内部用）",
                  "D": "D 类 · 被取代的早期批次",
                  "E": "E 类 · 冒烟测试",
                  "F": "F 类 · 出错或设计作废"}
        for k in "ABCDEF":
            v = by_cls.get(k, [])
            if not v:
                continue
            L += [f"## {titles[k]}　（{len(v)} 个）", "",
                  "| run_id | 调用 | 失败 | 臂 | 数据集 | 时间 | 说明 |",
                  "| --- | ---: | ---: | ---: | --- | --- | --- |"]
            for i in sorted(v, key=lambda x: x["name"]):
                arms = len(i.get("arms_seen") or i.get("arms") or {})
                L.append(f"| `{i['name']}` | {i['n_raw']:,} | {i['n_fail'] or ''} | "
                         f"{arms} | {i.get('dataset', '?')} | {i['mtime']} | {i['why']} |")
            L.append("")
        L += ["## 失败请求明细", "",
              "| run_id | 失败 | 总数 | 比例 | 错误类型 |", "| --- | ---: | ---: | ---: | --- |"]
        for i in bad:
            L.append(f"| `{i['name']}` | {i['n_fail']} | {i['n_raw']} | "
                     f"{i['n_fail'] / i['n_raw']:.1%} | {i['errors']} |")
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n  已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
