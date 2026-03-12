@echo off
setlocal
cd /d "%~dp0"

echo Running full Falcon H1R ablation test...
set VENV_PYTHON=.venv\Scripts\python.exe
set PYTHONPATH=%CD%\src

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found.
    pause
    exit /b 1
)

REM Run full Heretic ablation with test config
"%VENV_PYTHON%" -m heretic.main --config config.test.toml

echo Test completed.
pause
endlocal
