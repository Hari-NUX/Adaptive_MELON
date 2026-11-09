param()

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot

try {
    & py -3.11 --version | Out-Null
}
catch {
    $python311Missing = $true
}
if ($LASTEXITCODE -ne 0) { $python311Missing = $true }
if ($python311Missing) {
    $wingetCommand = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wingetCommand) {
        throw 'Python 3.11 is required for Windows PyTorch. Install it from https://www.python.org/downloads/ and rerun this command.'
    }
    $reply = Read-Host 'Python 3.11 is not installed. Allow winget to install it for this Windows user? [y/N]'
    if ($reply -notmatch '^(y|yes)$') { throw 'Python installation declined; no model checkpoint was downloaded.' }
    & winget install --id Python.Python.3.11 --exact --scope user --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 installation failed. Install it manually, reopen PowerShell, and rerun.' }
    & py -3.11 --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 was installed but is not visible to the py launcher yet. Reopen PowerShell and rerun.' }
}

$gmailRaw = Join-Path $projectRoot 'datasets\gmail\raw'
$expectedFiles = @('email_train.jsonl', 'email_test.jsonl', 'text_attack_train.json', 'text_attack_test.json')
$missing = @($expectedFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $gmailRaw $_)) })
if ($missing.Count -gt 0) {
    $reply = Read-Host 'BIPIA email files are missing. Download the four email files now? [y/N]'
    if ($reply -notmatch '^(y|yes)$') {
        throw 'BIPIA email data is required for the Gmail phase. No internet data will be downloaded.'
    }
    & (Join-Path $projectRoot 'scripts\download_datasets.ps1') -SkipBrowseSafe
}

& py -3.11 (Join-Path $projectRoot 'scripts\audit_bipia_email.py')
if ($LASTEXITCODE -ne 0) { throw 'BIPIA email audit failed.' }
& py -3.11 (Join-Path $projectRoot 'scripts\prepare_bipia_scenario_sample.py')
if ($LASTEXITCODE -ne 0) { throw 'BIPIA scenario sample preparation failed.' }
& py -3.11 (Join-Path $projectRoot 'scripts\build_gmail_action_scenarios.py')
if ($LASTEXITCODE -ne 0) { throw 'Gmail action scenario generation failed.' }

$trainLabels = Join-Path $projectRoot 'datasets\gmail\prepared\melon_teacher_train.jsonl'
$validationLabels = Join-Path $projectRoot 'datasets\gmail\prepared\melon_teacher_validation.jsonl'
$unreviewedTraces = Join-Path $projectRoot 'datasets\gmail\prepared\gmail_teacher_traces_unreviewed.jsonl'
$traceGenerator = Join-Path $projectRoot 'scripts\generate_teacher_traces.py'
$tracePartitioner = Join-Path $projectRoot 'scripts\partition_teacher_traces.py'
if (-not (Test-Path -LiteralPath $trainLabels) -and -not (Test-Path -LiteralPath $validationLabels)) {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw 'Ollama is required for local teacher-trace generation. Install/start Ollama, then rerun this command.'
    }
    & ollama show qwen3:1.7b *> $null
    if ($LASTEXITCODE -ne 0) {
        $teacherReply = Read-Host 'Local teacher model qwen3:1.7b is missing. Allow Ollama to download it? [y/N]'
        if ($teacherReply -notmatch '^(y|yes)$') { throw 'Teacher model download declined; no Hugging Face model was downloaded.' }
        & ollama pull qwen3:1.7b
        if ($LASTEXITCODE -ne 0) { throw 'Could not download the local Ollama teacher model.' }
    }
    Write-Host 'Generating paired traces for 250 Gmail scenarios with local Ollama qwen3:1.7b.'
    Write-Host 'This may take a while on CPU. No real email actions are possible in the simulator.'
    & py -3.11 $traceGenerator --model qwen3:1.7b --resume
    if ($LASTEXITCODE -ne 0) { throw 'Paired teacher-trace generation failed.' }
    & py -3.11 $tracePartitioner --input $unreviewedTraces
    if ($LASTEXITCODE -ne 0) { throw 'Teacher-trace partitioning failed.' }
}
elseif (-not (Test-Path -LiteralPath $trainLabels) -or -not (Test-Path -LiteralPath $validationLabels)) {
    throw 'Only one teacher partition exists. Review/repair the data files before training; no files were overwritten.'
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & py -3.11 -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the project environment.' }
& $venvPython -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2'
if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU PyTorch.' }
& $venvPython -m pip install -e $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not install the project training dependencies.' }
& $venvPython -m melon_guard
exit $LASTEXITCODE
