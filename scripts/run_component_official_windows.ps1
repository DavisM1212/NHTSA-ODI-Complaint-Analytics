param(
    [ValidateSet("CPU", "GPU", "cpu", "gpu")]
    [string]$TaskType = "CPU",
    [string]$Devices = "0",
    [ValidateSet("parquet", "csv")]
    [string]$OutputFormat = "parquet",
    [switch]$OverwriteExtracted,
    [switch]$SkipIngest,
    [switch]$SkipVisuals
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

Write-Host "NHTSA ODI Complaint Analytics - Official component pipeline"
Write-Host "Repository root: $RepoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Task type: $TaskType"

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
Invoke-Module "src.modeling.component_single_text_calibrated" @("--task-type", $TaskType, "--devices", $Devices)
Invoke-Module "src.modeling.component_multi_routing" @("--task-type", $TaskType, "--devices", $Devices)
Invoke-Module "src.reporting.update_component_readme"

if (-not $SkipVisuals) {
    Invoke-Module "src.reporting.component_visuals"
}

Write-Host ""
Write-Host "Official component pipeline completed"
Write-Host "Check data/processed/, data/outputs/, and docs/figures/component_models/"
