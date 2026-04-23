param(
    [ValidateSet("parquet", "csv")]
    [string]$OutputFormat = "parquet",
    [switch]$OverwriteExtracted,
    [switch]$SkipIngest,
    [int]$RandomSeed = 42,
    [string]$PublishStatus = "official"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-Module {
    param(
        [string]$ModuleName,
        [string[]]$Arguments = @()
    )

    Write-Step "Running $ModuleName"
    & $PythonExe -m $ModuleName @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $ModuleName"
        exit $LASTEXITCODE
    }
}

$PythonExe = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "NHTSA ODI Complaint Analytics - Official severity pipeline"
Write-Host "Repository root: $RepoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Random seed: $RandomSeed"

Write-Step "Running install verification"
& $PythonExe scripts\\verify_install.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install verification failed"
    exit $LASTEXITCODE
}

if (-not $SkipIngest) {
    $ingestArgs = @("--output-format", $OutputFormat)
    if ($OverwriteExtracted) {
        $ingestArgs += "--overwrite-extracted"
    }
    Invoke-Module "src.data.ingest_odi" $ingestArgs
}

Invoke-Module "src.preprocessing.clean_complaints"
Invoke-Module "src.modeling.severity_urgency_model" @("--random-seed", $RandomSeed, "--publish-status", $PublishStatus)

Write-Host ""
Write-Host "Official severity pipeline completed"
Write-Host "Check data/processed/ and data/outputs/"
