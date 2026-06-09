# ==============================================================================
# AMEVA-LLM-Trainer Windows PowerShell Setup Script
# ==============================================================================

Write-Host "=" * 60
Write-Host "   AMEVA-LLM-Trainer Windows Environment Setup"
Write-Host "=" * 60
Write-Host ""

$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ROOT) { $ROOT = (Get-Location).Path }
Set-Location $ROOT

# 1. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "[INFO] Creating virtual environment (venv)..."
    python -m venv venv
} else {
    Write-Host "[INFO] Virtual environment already exists."
}

# 2. Activate venv
Write-Host "[INFO] Activating virtual environment..."
& "venv\Scripts\Activate.ps1"

# 3. Upgrade pip
Write-Host "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

# 4. Install dependencies
Write-Host "[INFO] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Create necessary directories
$dirs = @("dataset", "outputs", "logs", "models\gguf", "configs")
foreach ($d in $dirs) {
    $fullPath = Join-Path $ROOT $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "[INFO] Created directory: $d"
    }
}

# 6. Create HF_HOME cache directory
$hfHome = "C:\ameva\models\llm"
if (-not (Test-Path $hfHome)) {
    New-Item -ItemType Directory -Path $hfHome -Force | Out-Null
    Write-Host "[INFO] Created HF_HOME cache directory: $hfHome"
}

# 7. Set environment variable
[System.Environment]::SetEnvironmentVariable("HF_HOME", $hfHome, "User")
Write-Host "[INFO] HF_HOME environment variable set to: $hfHome"

Write-Host ""
Write-Host "[SUCCESS] Setup process completed successfully!"
Write-Host "To run the application, execute: run_cli.bat"
Write-Host ""
