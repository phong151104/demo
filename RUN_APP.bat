@echo off
echo ========================================
echo   Credit Scoring System - Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [SUCCESS] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/Update requirements
echo [INFO] Installing/Updating dependencies...
pip install -r requirements.txt --quiet
echo [SUCCESS] Dependencies installed
echo.

REM Run Streamlit app
echo [INFO] Starting Credit Scoring System...
echo [INFO] The app will open in your default browser
echo [INFO] Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause

