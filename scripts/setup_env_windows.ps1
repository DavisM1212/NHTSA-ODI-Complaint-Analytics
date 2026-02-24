param(
    [string]$TargetPythonVersion = "3.13.12"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Get-VersionText {
    param([scriptblock]$CommandBlock)
    try {
        $output = & $CommandBlock 2>&1 | Out-String
        return $output.Trim()
    }
    catch {
        return $null
    }
}

function Parse-PythonVersion {
    param([string]$VersionText)
    if (-not $VersionText) {
        return $null
    }
    $match = [regex]::Match($VersionText, "Python\s+(\d+\.\d+\.\d+)")
    if ($match.Success) {
        return $match.Groups[1].Value
    }
    return $null
}

function Get-PythonSelector {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $py313Version = Parse-PythonVersion (Get-VersionText { py -3.13 --version })
        if ($py313Version) {
            return @{
                Label = "py -3.13"
                Version = $py313Version
                Runner = { param([string[]]$CommandArgs) & py -3.13 @CommandArgs }
            }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonVersion = Parse-PythonVersion (Get-VersionText { python --version })
        if ($pythonVersion) {
            return @{
                Label = "python"
                Version = $pythonVersion
                Runner = { param([string[]]$CommandArgs) & python @CommandArgs }
            }
        }
    }

    return $null
}

function Ensure-Python313 {
    param([string]$TargetVersion)

    $selector = Get-PythonSelector
    if ($selector -and $selector.Version -like "3.13.*") {
        return $selector
    }

    Write-Host "Python 3.13 was not detected"
    Write-Step "Attempting Python install via winget (Python.Python.3.13)"

    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wingetCmd) {
        Write-Host "winget not found"
        return $null
    }

    try {
        winget install --id Python.Python.3.13 -e --accept-package-agreements --accept-source-agreements
    }
    catch {
        Write-Host "winget install failed: $($_.Exception.Message)"
        return $null
    }

    return Get-PythonSelector
}

Write-Host "NHTSA ODI Complaint Analytics - Windows environment setup"
Write-Host "Repository root: $RepoRoot"

Write-Step "Checking Python"
$pythonSelector = Ensure-Python313 -TargetVersion $TargetPythonVersion

if (-not $pythonSelector) {
    Write-Host ""
    Write-Host "Automatic install failed or Python 3.13 is still unavailable"
    Write-Host "Manual install steps"
    Write-Host "1) Install Python $TargetPythonVersion (or latest Python 3.13.x) from python.org"
    Write-Host "2) Re-open PowerShell"
    Write-Host "3) Re-run .\\scripts\\setup_env_windows.ps1"
    exit 1
}

Write-Host "Using interpreter selector: $($pythonSelector.Label)"
Write-Host "Detected version: $($pythonSelector.Version)"
if ($pythonSelector.Version -ne $TargetPythonVersion) {
    Write-Host "Warning: recommended version is $TargetPythonVersion, but continuing with $($pythonSelector.Version)"
}

Write-Step "Creating virtual environment (.venv) if needed"
$VenvPython = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $pythonSelector["Runner"] @("-m", "venv", ".venv")
}
if (-not (Test-Path $VenvPython)) {
    Write-Host "Failed to create .venv"
    exit 1
}

Write-Step "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

Write-Step "Installing requirements.txt"
& $VenvPython -m pip install -r requirements.txt

Write-Step "Running install verification"
& $VenvPython scripts\\verify_install.py
$verifyExitCode = $LASTEXITCODE

Write-Host ""
if ($verifyExitCode -eq 0) {
    Write-Host "Setup completed successfully"
    Write-Host "Next step: .\\scripts\\run_pipeline_windows.ps1"
}
else {
    Write-Host "Setup completed with verification failures"
    Write-Host "Review the messages above, fix the issues, then rerun this script"
}

Write-Host "Tip: Activate the venv in a new shell with .\\.venv\\Scripts\\Activate.ps1"
exit $verifyExitCode
