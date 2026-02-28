@echo off
REM Medical Chatbot Backend Startup Script
REM Run this script from the MedicalChatbot directory

echo ========================================
echo Medical Chatbot Backend - Starting...
echo Current Directory: %CD%
echo ========================================

REM Check if virtual environment exists
if not exist ".venv\Scripts\Activate.bat" (
    echo Virtual environment not found! Creating one...
    python -m venv .venv
    echo Virtual environment created. Installing dependencies...
    call .venv\Scripts\pip install -r requirements.txt
)

REM Set PYTHONPATH to ensure imports work correctly
set PYTHONPATH=%CD%

echo Starting Uvicorn server...
echo Server will be available at: http://127.0.0.1:8000
echo API Documentation: http://127.0.0.1:8000/docs
echo ========================================

REM Run uvicorn
call .venv\Scripts\uvicorn app.main:app --reload --port 8000
