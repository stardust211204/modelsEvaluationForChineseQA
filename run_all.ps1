# 一键复现：小样本走完整条管线
#
#   .\run_all.ps1                 # 小样本（5 题）全流程演练
#   .\run_all.ps1 -N 20           # 20 题
#   .\run_all.ps1 -Reps 2         # 难度代理每题采样 2 次
#
# 用途：干净目录复现演练（见 docs/pipeline.md）。不做全量跑批 —— 全量见 README。

param(
    [int]$N = 5,
    [int]$Reps = 2,
    [switch]$SkipProbes
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "!! 未找到虚拟环境，请先执行：" -ForegroundColor Red
    Write-Host "   uv venv .venv --python 3.12"
    Write-Host "   uv pip install --python `"$py`" openai datasets pandas scipy matplotlib pyyaml tqdm pytest"
    exit 1
}

$env:PYTHONIOENCODING = "utf-8"
$env:HF_ENDPOINT = "https://hf-mirror.com"

function Step($n, $title) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host "  $n  $title" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor Cyan
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# ---- 0. 单测（判分逻辑是唯一能拦住静默错误的地方，不能跳）------------------
Step "0/7" "单元测试"
& $py -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) { Write-Host "!! 单测未通过，中止" -ForegroundColor Red; exit 1 }

# ---- 1. 三臂连通性 ---------------------------------------------------------
if (-not $SkipProbes) {
    Step "1/7" "三臂连通性冒烟"
    & $py scripts\smoke_arms.py --n 1
    if ($LASTEXITCODE -ne 0) { Write-Host "!! 连通性冒烟未通过（可能存在空 content）" -ForegroundColor Yellow }
} else {
    Step "1/7" "跳过探针（-SkipProbes）"
}

# ---- 2. 造题 ---------------------------------------------------------------
Step "2/7" "造题（分层抽样 + 冻结）"
& $py scripts\build_set.py --n 600

# ---- 3. 难度代理（可选，本地模型）-----------------------------------------
Step "3/7" "难度代理（本地 $N 题 × $Reps 次）"
try {
    & $py scripts\difficulty_proxy.py --reps $Reps --n $N
} catch {
    Write-Host "  难度代理跳过（本地模型不可用不影响主流程）" -ForegroundColor Yellow
}

# ---- 4. 端到端管线（含断点续跑验证）--------------------------------------
Step "4/7" "端到端管线（造题→构造→执行→判分，含续跑验证）"
& $py scripts\smoke_pipeline.py --n $N
if ($LASTEXITCODE -ne 0) { Write-Host "!! 管线演练未通过" -ForegroundColor Red; exit 1 }

# ---- 5. 产物清点 -----------------------------------------------------------
Step "5/7" "产物清点"
foreach ($p in @("data\eval_set.jsonl", "data\difficulty_proxy.json", "runs\pipe-smoke\manifest.json")) {
    $full = Join-Path $root $p
    if (Test-Path $full) {
        $kb = [math]::Round((Get-Item $full).Length / 1KB, 1)
        Write-Host ("  [OK] {0,-44} {1,8} KB" -f $p, $kb) -ForegroundColor Green
    } else {
        Write-Host ("  [--] {0,-44} 未生成" -f $p) -ForegroundColor DarkGray
    }
}

# ---- 6. 回归门禁 -----------------------------------------------------------
Step "6/7" "回归门禁（题集冻结 / 参数冻结 / 判分口径零分歧）"
& $py scripts\ci_gate.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "!! 门禁未通过（见上方 FAIL 项）—— 结论级前提被破坏，不要据此出报告" -ForegroundColor Red
}

$sw.Stop()
Step "7/7" "完成"
Write-Host ("  总耗时 {0:N0} 秒" -f $sw.Elapsed.TotalSeconds)
Write-Host "  下一步：全量跑批见 README「快速开始」" -ForegroundColor Green
