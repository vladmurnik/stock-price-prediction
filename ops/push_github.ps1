# Publish this repository to GitHub.
#
# Nothing here is executed automatically: the project has no remote and no credentials in the
# environment, so publishing is a deliberate, user-run step.
#
# Usage:
#   $env:GITHUB_TOKEN = "ghp_..."          # scopes: repo (classic) or Contents:write (fine-grained)
#   .\ops\push_github.ps1 -RepoUrl "https://github.com/<user>/<repo>.git"
#
# The token is read from the environment, passed to git over stdin-style credential config for
# this one push only, and never written to disk.

param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $env:GITHUB_TOKEN) {
    throw "GITHUB_TOKEN is not set. Create a token with the 'repo' scope and set the variable first."
}

git rev-parse --is-inside-work-tree | Out-Null

$remotes = git remote
if ($remotes -notcontains "origin") {
    git remote add origin $RepoUrl
} else {
    git remote set-url origin $RepoUrl
}

$clean = (git status --porcelain | Measure-Object).Count
if ($clean -gt 0) {
    Write-Host "warning: working tree has $clean uncommitted change(s); pushing the last commit only"
}

# Feed the credential through a temporary askpass helper so the token never lands in .git/config.
$askpass = Join-Path $env:TEMP "spp-askpass.cmd"
@(
    "@echo off",
    "echo %GITHUB_TOKEN%"
) | Set-Content -Path $askpass -Encoding ASCII

try {
    $env:GIT_ASKPASS = $askpass
    $env:GIT_TERMINAL_PROMPT = "0"
    git push -u origin $Branch
} finally {
    Remove-Item $askpass -ErrorAction SilentlyContinue
    Remove-Item Env:GIT_ASKPASS -ErrorAction SilentlyContinue
}

Write-Host "pushed $Branch to $RepoUrl"
