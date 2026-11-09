param(
    [string]$Model = 'qwen3:1.7b'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot

Write-Host 'Gmail data preparation only: scenario build, paired local traces, and leakage-checked partitioning.'
Write-Host 'This does not install Python packages, download Hugging Face checkpoints, or start training.'
Write-Host "Project: $projectRoot"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) { throw 'Python was not found on PATH. Install Python and reopen PowerShell.' }
& python --version
if ($LASTEXITCODE -ne 0) { throw 'Could not run Python.' }

$requiredFiles = @(
    'datasets\gmail\raw\email_train.jsonl',
    'datasets\gmail\raw\email_test.jsonl',
    'datasets\gmail\raw\text_attack_train.json',
    'datasets\gmail\raw\text_attack_test.json'
)
$missing = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $projectRoot $_)) })
if ($missing.Count -gt 0) { throw "Missing BIPIA email files: $($missing -join ', '). Put all four files in datasets\gmail\raw\ and rerun." }

Write-Host "`n[1/4] Auditing the four BIPIA email source files..."
& python (Join-Path $projectRoot 'scripts\audit_bipia_email.py')
if ($LASTEXITCODE -ne 0) { throw 'BIPIA source audit failed; fix the reported issue before continuing.' }

$scenarioFile = Join-Path $projectRoot 'datasets\gmail\prepared\gmail_action_scenarios.jsonl'
if (-not (Test-Path -LiteralPath $scenarioFile)) {
    Write-Host "`n[2/4] Building training-only simulated Gmail action scenarios..."
    & python (Join-Path $projectRoot 'scripts\build_gmail_action_scenarios.py')
    if ($LASTEXITCODE -ne 0) { throw 'Scenario generation failed.' }
} else {
    Write-Host "`n[2/4] Reusing the existing scenario file: $scenarioFile"
}

$traceFile = Join-Path $projectRoot 'datasets\gmail\prepared\gmail_teacher_traces_unreviewed.jsonl'
$trainFile = Join-Path $projectRoot 'datasets\gmail\prepared\melon_teacher_train.jsonl'
$validationFile = Join-Path $projectRoot 'datasets\gmail\prepared\melon_teacher_validation.jsonl'
if ((Test-Path -LiteralPath $trainFile) -or (Test-Path -LiteralPath $validationFile)) {
    throw 'A teacher partition already exists. Move it aside for safekeeping before regenerating partitions.'
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw 'Ollama CLI was not found. Start/install Ollama, then rerun this command.' }
& ollama show $Model *> $null
if ($LASTEXITCODE -ne 0) { throw "Ollama model '$Model' is not installed locally. Pull it separately, or rerun with -Model <installed-tag>." }

Write-Host "`n[3/4] Generating paired original/replay teacher traces with local Ollama model '$Model'."
Write-Host 'Rows are saved incrementally. If you stop the command, run this same command again; it resumes missing scenarios.'
Write-Host 'All actions take place in the local simulator; no real email is sent.'
& python (Join-Path $projectRoot 'scripts\generate_teacher_traces.py') --model $Model --resume
if ($LASTEXITCODE -ne 0) { throw 'Trace generation stopped or failed. The saved rows are retained; rerun this same command to resume.' }

Write-Host "`n[4/4] Creating train/validation files, grouped by source email..."
& python (Join-Path $projectRoot 'scripts\partition_teacher_traces.py') --input $traceFile
if ($LASTEXITCODE -ne 0) { throw 'Partitioning failed. The unreviewed traces remain available for inspection.' }

Write-Host "`nGmail preparation is complete. Inspect these files before training:"
Write-Host "  $traceFile"
Write-Host "  $trainFile"
Write-Host "  $validationFile"
Write-Host 'No training has been started.'
