param(
    [switch]$SkipBIPIA,
    [switch]$SkipBrowseSafe
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Save-VerifiedDownload([string]$Url, [string]$Destination) {
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $partial = "$Destination.partial"
    if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    Write-Host "Downloading $(Split-Path -Leaf $Destination)"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $partial
        $file = Get-Item -LiteralPath $partial
        if ($file.Length -lt 20) { throw "Downloaded file is unexpectedly small ($($file.Length) bytes)." }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
    }
    catch {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        throw "Download failed: $Url`n$($_.Exception.Message)"
    }
}

if (-not $SkipBIPIA) {
    $gmail = Join-Path $repoRoot 'datasets\gmail\raw'
    $bipiaFiles = @{
        'email_train.jsonl' = 'https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/train.jsonl'
        'email_test.jsonl' = 'https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl'
        'text_attack_train.json' = 'https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_train.json'
        'text_attack_test.json' = 'https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json'
    }
    foreach ($name in $bipiaFiles.Keys) {
        Save-VerifiedDownload $bipiaFiles[$name] (Join-Path $gmail $name)
    }
}

if (-not $SkipBrowseSafe) {
    $internet = Join-Path $repoRoot 'datasets\internet\raw'
    $browseSafeFiles = @{
        'browsesafe_train.parquet' = 'https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/train.parquet'
        'browsesafe_test.parquet' = 'https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/test.parquet'
    }
    foreach ($name in $browseSafeFiles.Keys) {
        Save-VerifiedDownload $browseSafeFiles[$name] (Join-Path $internet $name)
    }
}

Write-Host "Dataset downloads complete under $(Join-Path $repoRoot 'datasets')"
Write-Host 'Check upstream license notes before redistribution. Test files are for evaluation only.'
