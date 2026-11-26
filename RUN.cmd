@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto virtual
python -c "import streamlit, sklearn, xgboost, torch, ultralytics, PIL; assert sklearn.__version__ == '1.8.0' and ultralytics.__version__ == '8.4.183' and xgboost.__version__ == '3.4.1'" >nul 2>nul
if not errorlevel 1 goto installed
python -m venv .venv
if errorlevel 1 goto failed
:virtual
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
if errorlevel 1 goto failed
exit /b 0
:installed
python -m streamlit run app.py --server.address localhost
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or startup failed. Check the message above and confirm that Python is installed.
pause
exit /b 1
