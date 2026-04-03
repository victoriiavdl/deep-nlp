@echo off
setlocal

:: Find uv (same search order as setup.bat)
set "UV="

where uv >nul 2>&1
if %errorlevel%==0 ( set "UV=uv" & goto :run )

if exist "%LOCALAPPDATA%\uv\uv.exe" (
    set "UV=%LOCALAPPDATA%\uv\uv.exe" & goto :run )

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV=%USERPROFILE%\.local\bin\uv.exe" & goto :run )

if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
    set "UV=%USERPROFILE%\.cargo\bin\uv.exe" & goto :run )

:: uv not found -- use .venv directly
if exist ".venv\Scripts\python.exe" (
    echo [uv not found, using .venv\Scripts\python.exe]
    .venv\Scripts\python.exe %*
    goto :end
)

echo [ERROR] Neither uv nor .venv found. Run setup.bat first.
goto :end

:: Run with uv
:run
"%UV%" run python %*

:end
endlocal
