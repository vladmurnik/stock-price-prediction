#requires -Version 5.1
<#
.SYNOPSIS
  Runs the next ready task from ops/queue.json with the assigned agent and records the report.

.EXAMPLES
  .\ops\dispatch.ps1 -DryRun
  .\ops\dispatch.ps1 -MaxTasks 1
  .\ops\dispatch.ps1 -Loop -MaxTasks 50
  .\ops\dispatch.ps1 -Only t06
#>
param(
    [int]$MaxTasks = 1,
    [switch]$Loop,
    [switch]$DryRun,
    [string]$Only = "",
    [int]$PauseSec = 3,
    [string]$Model = "freepi/deepseek-v4.1-flash"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$queuePath  = Join-Path $root "ops\queue.json"
$inboxDir   = Join-Path $root "ops\inbox"
$statusFile = Join-Path $root "ops\status.md"

New-Item -ItemType Directory -Force -Path $inboxDir | Out-Null
if (-not (Test-Path $statusFile)) {
    [System.IO.File]::WriteAllText($statusFile,
        "| время | задача | агент | статус | отчёт |`r`n|---|---|---|---|---|`r`n",
        (New-Object System.Text.UTF8Encoding($false)))
}

function Load-Queue {
    $raw = [System.IO.File]::ReadAllText($queuePath, [System.Text.Encoding]::UTF8)
    return $raw | ConvertFrom-Json
}

function Save-Queue($q) {
    $json = $q | ConvertTo-Json -Depth 12
    $tmp = "$queuePath.tmp"
    [System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -Force $tmp $queuePath
}

function Get-NextTask($q) {
    $done = @{}
    foreach ($t in $q.tasks) { if ($t.status -eq "done") { $done[$t.id] = $true } }
    foreach ($t in $q.tasks) {
        if ($t.status -ne "pending") { continue }
        $ok = $true
        foreach ($d in @($t.depends_on)) {
            if ($d -and -not $done.ContainsKey($d)) { $ok = $false; break }
        }
        if ($ok) { return $t }
    }
    return $null
}

function Get-StatusFromReport([string]$text) {
    if ($text -match "(?m)^##\s*STATUS:\s*AUDIT-COMPLETE") { return "done" }
    if ($text -match "(?m)^##\s*STATUS:\s*DONE")          { return "done" }
    if ($text -match "(?m)^##\s*STATUS:\s*BLOCKED")       { return "blocked" }
    if ($text -match "(?m)^##\s*STATUS:\s*NEEDS-DECISION"){ return "needs-decision" }
    return "needs-decision"
}

function Add-StatusRow([string]$ts, [string]$id, [string]$agent, [string]$status, [string]$report) {
    $rel = $report.Replace($root + "\", "")
    $row = "| $ts | $id | $agent | $status | $rel |`r`n"
    [System.IO.File]::AppendAllText($statusFile, $row, (New-Object System.Text.UTF8Encoding($false)))
}

$executed = 0

while ($executed -lt $MaxTasks) {
    $q = Load-Queue

    if ($Only) {
        $task = $q.tasks | Where-Object { $_.id -eq $Only -and $_.status -eq "pending" } | Select-Object -First 1
        if (-not $task) { Write-Host "Task '$Only' is not pending."; break }
    }
    else {
        $task = Get-NextTask $q
    }

    if (-not $task) {
        Write-Host "No ready tasks. Done=$(($q.tasks | Where-Object status -eq 'done').Count)/$($q.tasks.Count)"
        break
    }

    Write-Host ("-> {0} [{1}] {2}" -f $task.id, $task.agent, $task.plan)
    if ($DryRun) {
        Write-Host "--- prompt ---"
        Write-Host $task.prompt
        break
    }

    $task.status = "running"
    $task | Add-Member -NotePropertyName started_at -NotePropertyValue (Get-Date).ToString("o") -Force
    Save-Queue $q

    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $reportPath = Join-Path $inboxDir "$ts-$($task.id)-$($task.agent).md"
    $out = ""
    $err = ""

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = -1
    try {
        $out = (& opencode run --model $Model --agent $task.agent --auto --title "$($task.id)-$($task.agent)" $task.prompt 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
    }
    catch {
        $err = $_.Exception.Message
        $out = ""
        $exitCode = -1
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
    if ($out) { $out = $out -replace "`e\[[0-9;]*m", "" }

    if ([string]::IsNullOrWhiteSpace($out)) {
        $out = "## STATUS: NEEDS-DECISION`r`n`r`nAgent returned empty output. Exit code: $exitCode`r`n`r`nDispatch error: $err"
    }

    $header = "# $($task.id) | $($task.agent) | $ts`r`n`r`n"
    [System.IO.File]::WriteAllText($reportPath, $header + $out, (New-Object System.Text.UTF8Encoding($false)))

    $status = Get-StatusFromReport $out

    # reload to avoid clobbering writes made by the agent itself
    $q = Load-Queue
    $persisted = $q.tasks | Where-Object { $_.id -eq $task.id } | Select-Object -First 1
    $persisted.status = $status
    $persisted | Add-Member -NotePropertyName finished_at -NotePropertyValue (Get-Date).ToString("o") -Force
    $persisted | Add-Member -NotePropertyName report -NotePropertyValue ("ops/inbox/" + (Split-Path $reportPath -Leaf)) -Force
    Save-Queue $q

    Add-StatusRow $ts $task.id $task.agent $status $reportPath
    Write-Host ("   {0} -> {1}" -f $task.id, $status)

    # Surface anything the worker wrote to the message bus during this run, then ack it.
    $bus = Join-Path $root "ops\bus.py"
    $busPy = Join-Path $root ".venv\Scripts\python.exe"
    if ((Test-Path $bus) -and (Test-Path $busPy)) {
        $busOut = & $busPy $bus read --to orchestrator --unread --limit 50 2>$null
        if ($busOut -and $busOut -notmatch '^no messages') {
            Write-Host "   --- bus ---"
            $busOut | ForEach-Object { Write-Host "   $_" }
            $ids = ($busOut | Select-String -Pattern '^\* #(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value })
            foreach ($id in $ids) { & $busPy $bus ack --id $id 2>$null | Out-Null }
            & $busPy $bus render 2>$null | Out-Null
        }
    }

    $executed++

    if ($status -eq "blocked" -or $status -eq "needs-decision") {
        Write-Host "Stopping: task requires attention. See $reportPath"
        break
    }
    if (-not $Loop) { break }
    if ($PauseSec -gt 0) { Start-Sleep -Seconds $PauseSec }
}

Write-Host ""
Write-Host "Executed: $executed"
