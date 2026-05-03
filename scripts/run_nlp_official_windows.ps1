param(
    [ValidateSet("parquet", "csv")]
    [string]$OutputFormat = "parquet",
    [switch]$OverwriteExtracted,
    [switch]$SkipIngest,
    [switch]$SkipCacheRebuild,
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

Write-Host "NHTSA ODI Complaint Analytics - Official NLP early-warning pipeline"
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

$nlpArgs = @("--random-seed", $RandomSeed, "--publish-status", $PublishStatus)
if ($SkipCacheRebuild) {
    $nlpArgs += "--skip-cache-rebuild"
}

Invoke-Module "src.modeling.nlp_early_warning_system" $nlpArgs
Invoke-Module "src.reporting.watchlist_visuals"

Write-Host ""
Write-Host "Official NLP early-warning pipeline completed"
Write-Host "Check data/processed/odi_nlp_prepped.parquet, data/outputs/, and docs/figures/nlp_early_warning/"
