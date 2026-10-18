"""把全部模型原始输出打包成一个 zip，用于挂 GitHub Release。

背景：报告摘要写着「模型原始输出数据均公开在 GitHub」。但 raw 有 75 MB，
      **不适合进 git 历史**（`.gitignore` 已排除 `runs/*/raw/`），
      所以走 Release 附件。本脚本负责把包做出来，上传仍需人工（见文件末尾提示）。

用法：
    python scripts/pack_raw_release.py            # 输出到 dist/raw-data.zip
    python scripts/pack_raw_release.py --out X.zip

包内结构（与仓库一致，解压即对得上）：
    runs/{run_id}/raw/{config}/{arm}.jsonl
    archive/runs/{run_id}/raw/{config}/{arm}.jsonl
    README.md          ← 本脚本生成，说明字段含义与复算方式
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

INNER_README = """# 模型原始输出（raw）

本包是 modelsEvaluation 项目**全部模型调用的原始落盘记录**，随 GitHub Release 分发 ——
它体积较大（约 75 MB），不适合进 git 历史，故不走仓库。

## 为什么它是「真值」

项目把 `raw` 与 `tallies` 物理分离：

- `raw/`（**本包**）：PromptSpec + ModelResult，**不可变**。每次调用的实际请求参数、
  完整回复、`finish_reason`、延迟、token 数都在这里。
- `tallies/`（**在仓库里**）：判分结果，**可重算**。

因此**改判分规则不需要重跑 API** —— 用 `tallies/` 就能复算报告里的全部指标，
不需要任何 API key；要换一套判分口径，用本包的 raw 重跑 `src/meval/tally.py` 即可。

## 目录结构

```
runs/{run_id}/raw/{config}/{arm}.jsonl          # 在用的权威批次
archive/runs/{run_id}/raw/{config}/{arm}.jsonl  # 归档批次（含被作废的首轮）
```

`run_id` 与 `reports/RUN_AUDIT.md`（全库 85 批次分类）、
`reports/RUN_INVENTORY.md`（37 个权威批次）一一对应。

## 每行 JSON 的主要字段

| 字段 | 含义 |
| --- | --- |
| `arm` | 模型名（被测模型） |
| `instance_id` | 题目 ID，可回连 `data/*.jsonl` |
| `rep` | 第几次重复（重复采样用） |
| `params` | **本次实际发出的参数** —— 参数是否生效以它为准，不以 manifest 为准 |
| `content` | 模型回复正文 |
| `reasoning` | 思维链内容（若厂商单独返回） |
| `finish_reason` | `stop` / `length`（后者计入 `truncated` 档） |
| `tokens_out` | 输出 token 数 |
| `latency_s` | 单次延迟（秒） |
| `error` | 传输层错误（成功记录该字段为空） |

## 复算

```powershell
# 判分台账已在仓库里，clone 后即可复算报告全部指标，无需 API key：
.venv\\Scripts\\python.exe scripts\\report_metrics.py
.venv\\Scripts\\python.exe scripts\\_check_polished_report.py

# 要用本包的 raw 重跑判分（换口径）：
.venv\\Scripts\\python.exe -m meval.tally --run <run_id>
```

数字来源与逐格复算命令见 `reports/report_data_provenance.md`；
口径定义见对外报告 §2.4。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/raw-data.zip")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for base in ("runs", "archive/runs"):
        d = ROOT / base
        if not d.exists():
            continue
        for raw in sorted(d.glob("*/raw")):
            files.extend(sorted(p for p in raw.rglob("*") if p.is_file()))

    if not files:
        print("  [失败] 没找到任何 raw 文件")
        return 1

    total = sum(p.stat().st_size for p in files)
    print("=" * 78)
    print("  打包模型原始输出（raw）")
    print("=" * 78)
    print(f"  文件数 : {len(files)}")
    print(f"  原始大小: {total / 1024 / 1024:.1f} MB")
    print(f"  产出   : {out.relative_to(ROOT)}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("README.md", INNER_README)
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(ROOT).as_posix())
            if i % 50 == 0:
                print(f"    ... {i}/{len(files)}")

    print(f"  压缩后 : {out.stat().st_size / 1024 / 1024:.1f} MB"
          f"（压缩率 {out.stat().st_size / total * 100:.0f}%）")
    print()
    print("  上传（需人工，本脚本不碰网络）：")
    print("    gh release create data-v1 " + args.out.replace("\\", "/") +
          " --title \"模型原始输出数据\" --notes \"全部模型调用的原始落盘记录…\"")
    print("  或：GitHub 仓库页 → Releases → Draft a new release → 拖入该 zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
