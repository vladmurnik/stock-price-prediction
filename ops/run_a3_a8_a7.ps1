# Close A3, A8, A7 after the open-jev corpus scoring finishes.
#
# Precondition: data\processed\article_sentiment_openjev.parquet exists
# (written by `python -m spp.cli score corpus --backend openjev`).
#
# RAM/CPU discipline: the steps run strictly sequentially, one heavy process at a time.
#
# ASCII only on purpose: Windows PowerShell 5.1 reads a BOM-less UTF-8 file as ANSI and a
# non-ASCII character inside a string literal breaks parsing with "UnexpectedToken".
#
# Usage:
#   .\ops\run_a3_a8_a7.ps1                # full run
#   .\ops\run_a3_a8_a7.ps1 -SkipBuild     # matrices already built

param(
    [switch]$SkipBuild,
    [switch]$SkipA3
)

# Keep this at "Continue": PowerShell 5.1 turns a child process writing to stderr into a
# NativeCommandError, and every LightGBM/torch warning goes to stderr. With "Stop" the pipeline
# aborts on the first deprecation warning (that is exactly how the a3 run died at line 58).
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "ops\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$sentiment = "data\processed\article_sentiment_openjev.parquet"
if (-not (Test-Path $sentiment)) {
    throw "missing $sentiment - open-jev scoring has not finished. Do not build A3/A8 without it."
}
$rows = & $py -c "import pandas as pd;print(len(pd.read_parquet(r'$sentiment')))"
Write-Host "open-jev sentiment: $rows rows"

function Step([string]$name, [string]$log, [scriptblock]$body) {
    Write-Host "=== $name ==="
    $started = Get-Date
    & $body *> (Join-Path $logs $log)
    $elapsed = (Get-Date) - $started
    Write-Host ("=== {0} done in {1:N1} min (log: ops/logs/{2})" -f $name, $elapsed.TotalMinutes, $log)
    # Inspect the output, not $LASTEXITCODE: PowerShell reports 1 when a child only writes
    # warnings to stderr (LightGBM, torch).
    $tail = Get-Content (Join-Path $logs $log) -Encoding UTF8 -Tail 40 |
        Where-Object { $_ -match "aggregate:|Traceback|SystemExit|built:|wrote|ERROR" }
    if ($tail) { $tail | ForEach-Object { Write-Host "  $_" } }
}

if (-not $SkipBuild) {
    Step "ablation matrices a3/a8" "a3a8_build.log" {
        & $py "ops\ablate.py" --only a3 a8
    }
}

if (-not $SkipA3) {
    Step "train a3-openjev (sentiment only)" "a3_train.log" {
        & $py -m spp.cli train gbdt `
            --features "data\processed\ablation\a3-openjev-sent.parquet" `
            --run-id "a3-openjev"
    }
}

Step "train a8-openjev (market + sentiment + typed flags)" "a8_train.log" {
    & $py -m spp.cli train gbdt `
        --features "data\processed\ablation\a8-openjev.parquet" `
        --run-id "a8-openjev"
}

Step "a7 ensemble (finbert + open-jev)" "a7_ensemble.log" {
    & $py "ops\ensemble_a7.py"
}

Step "ablation report + results" "report.log" {
    & $py "ops\make_ablation_report.py"
    & $py "ops\collect_results.py"
}

Write-Host ""
Write-Host "done. Check results\ablation.md and reports\runs\{a3-openjev,a7-ensemble,a8-openjev}\metrics.json"
