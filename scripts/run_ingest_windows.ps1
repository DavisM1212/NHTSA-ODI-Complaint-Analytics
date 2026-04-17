param(
    [ValidateSet("parquet", "csv")]
    [string]$OutputFormat = "parquet",
    [switch]$OverwriteExtracted,
    [switch]$NoCombine
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

$PythonExe = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "NHTSA ODI Complaint Analytics - Windows ingest runner"
Write-Host "Repository root: $RepoRoot"
Write-Host "Python: $PythonExe"

Write-Step "Running install verification"
& $PythonExe scripts\\verify_install.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Install verification failed, stopping pipeline"
    exit $LASTEXITCODE
}

$moduleArgs = @("-m", "src.data.ingest_odi", "--output-format", $OutputFormat)
if ($OverwriteExtracted) {
    $moduleArgs += "--overwrite-extracted"
}
if ($NoCombine) {
    $moduleArgs += "--no-combine"
}

Write-Step "Running ODI complaint extraction + combine"
& $PythonExe @moduleArgs
$pipelineExitCode = $LASTEXITCODE

Write-Host ""
if ($pipelineExitCode -eq 0) {
    Write-Host "Ingest completed"
    Write-Host "Check data/extracted/ and data/processed/"
}
else {
    Write-Host "Ingest failed"
}

exit $pipelineExitCode
