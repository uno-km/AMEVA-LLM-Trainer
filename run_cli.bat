@echo off
echo ======================================================================
echo AMEVA-LLM-Trainer Premium CLI Launcher
echo ======================================================================

IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup.py first.
    pause
    exit /b
)

echo [INFO] Checking if backend API server is running on port 8001...
netstat -ano | find "8001" >nul
if %errorlevel% neq 0 (
    echo [WARN] API server is not running on 8001. Automatically starting the server...
    start "AMEVA LLM API Server" run_server.bat
    echo [INFO] Waiting 5 seconds for the server to initialize...
    timeout /t 5 /nobreak >nul
) else (
    echo [INFO] API server on 8001 is already running!
)

echo Starting AMEVA-LLM-Trainer CLI...
venv\Scripts\python.exe cli\cli.py
pause
