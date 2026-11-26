@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -c "import flask" >nul 2>nul
if errorlevel 1 (
    "%PYTHON%" -m pip install "Flask>=3.1,<4"
    if errorlevel 1 goto failed
)
echo Open http://127.0.0.1:8502 in your browser.
"%PYTHON%" -m overwatch_web.web
if errorlevel 1 goto failed
exit /b 0
:failed
pause
exit /b 1
