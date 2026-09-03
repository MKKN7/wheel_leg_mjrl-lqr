@echo off
setlocal
if not defined ROBOT7_PYTHON set "ROBOT7_PYTHON=python"

where "%ROBOT7_PYTHON%" >nul 2>&1
if errorlevel 1 if not exist "%ROBOT7_PYTHON%" (
    echo Python executable was not found: %ROBOT7_PYTHON%
    echo Set ROBOT7_PYTHON to your Python 3.12 environment before retrying.
    exit /b 1
)

if "%~1"=="" (
    echo Usage: robot7_python.cmd script.py [arguments]
    echo Example: robot7_python.cmd train_warp_ppo.py
    exit /b 2
)

pushd "%~dp0"
"%ROBOT7_PYTHON%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
