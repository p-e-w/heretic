@echo off
setlocal
cd /d "%~dp0"

echo Testing Falcon H1R loading...
set VENV_PYTHON=.venv\Scripts\python.exe
set PYTHONPATH=%CD%\src

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found.
    pause
    exit /b 1
)

REM Test loading with bfloat16 on CPU - should work on ROG Ally
"%VENV_PYTHON%" -m heretic.main --model tiiuae/Falcon-H1R-7B --dtypes bfloat16 --device-map cpu --trust-remote-code

echo Test completed.
pause
endlocal
