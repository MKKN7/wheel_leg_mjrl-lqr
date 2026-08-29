@echo off
setlocal
set "ROBOT7_PYTHON=F:\Users\ASUS\Anaconda3\envs\robot7\python.exe"

if not exist "%ROBOT7_PYTHON%" (
    echo robot7 Python was not found at:
    echo %ROBOT7_PYTHON%
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
