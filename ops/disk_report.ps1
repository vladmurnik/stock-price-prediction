param([double]$MinFreeGB = 8)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$drive = Get-PSDrive -Name ($root.Substring(0, 1))

function Get-FolderSizeGB([string]$path) {
    if (-not (Test-Path $path)) { return 0.0 }
    $bytes = (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if (-not $bytes) { return 0.0 }
    return [math]::Round($bytes / 1GB, 2)
}

$rows = @(
    [pscustomobject]@{ Item = 'venv (.venv)'; GB = (Get-FolderSizeGB (Join-Path $root '.venv')) }
    [pscustomobject]@{ Item = 'data (parquet)'; GB = (Get-FolderSizeGB (Join-Path $root 'data')) }
    [pscustomobject]@{ Item = 'reports'; GB = (Get-FolderSizeGB (Join-Path $root 'reports')) }
    [pscustomobject]@{ Item = 'HF hub cache'; GB = (Get-FolderSizeGB (Join-Path $env:USERPROFILE '.cache\huggingface\hub')) }
    [pscustomobject]@{ Item = 'pip cache'; GB = (Get-FolderSizeGB (Join-Path $env:LOCALAPPDATA 'pip\Cache')) }
)

$used = ($rows | Measure-Object -Property GB -Sum).Sum
$rows | Format-Table -AutoSize

Write-Host ("project footprint: {0:N2} GB" -f $used)
Write-Host ("disk {0}: free {1:N1} GB of {2:N1} GB" -f $drive.Name, ($drive.Free / 1GB), (($drive.Used + $drive.Free) / 1GB))

if ($drive.Free / 1GB -lt $MinFreeGB) {
    Write-Host "BLOCKED: free space below $MinFreeGB GB" -ForegroundColor Red
    exit 2
}

Write-Host 'OK'
exit 0
