@echo off
REM Fashion Recommendation Backend - Startup Script

echo ========================================
echo Fashion Recommendation Backend Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Start the server
echo ========================================
echo Starting FastAPI server...
echo ========================================
echo.
echo API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
