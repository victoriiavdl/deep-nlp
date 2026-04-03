@echo off
setlocal

echo ============================================================
echo   Financial Sentiment Analysis — Environment Setup
echo ============================================================
echo.

:: ── Find uv ───────────────────────────────────────────────────
set "UV="

where uv >nul 2>&1
if %errorlevel%==0 (
    set "UV=uv"
    goto :found_uv
)

if exist "%LOCALAPPDATA%\uv\uv.exe" (
    set "UV=%LOCALAPPDATA%\uv\uv.exe"
    goto :found_uv
)

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV=%USERPROFILE%\.local\bin\uv.exe"
    goto :found_uv
)

if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "UV=%USERPROFILE%\.cargo\bin\uv.exe"
    goto :found_uv
)

:: ── uv not found — fall back to pip ───────────────────────────
echo [!] uv not found. Falling back to pip.
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating .venv...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -e .

echo.
echo [!] IMPORTANT: pip installs CPU-only PyTorch by default.
echo     For GPU support, run this AFTER the install:
echo.
echo     pip install torch --index-url https://download.pytorch.org/whl/cu128
echo.
echo Setup complete (pip). Activate with: .venv\Scripts\activate
goto :end

:: ── uv found ──────────────────────────────────────────────────
:found_uv
echo [OK] Found uv at: %UV%
echo.

echo Syncing environment (this installs all dependencies including CUDA PyTorch)...
"%UV%" sync

if %errorlevel% neq 0 (
    echo [ERROR] uv sync failed. Check the output above.
    goto :end
)

echo.
echo ============================================================
echo   Setup complete.
echo ============================================================
echo.
echo To run commands, use:
echo.
echo   "%UV%" run python scripts/train_baselines.py
echo   "%UV%" run python scripts/train_transformer.py --all
echo   "%UV%" run python scripts/build_leaderboard.py
echo   "%UV%" run python app.py
echo.

:end
endlocal
