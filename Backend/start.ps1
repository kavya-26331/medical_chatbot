# Medical Chatbot Backend Startup Script
# Run this script from the MedicalChatbot directory

$ErrorActionPreference = "Stop"

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Medical Chatbot Backend - Starting..." -ForegroundColor Cyan
Write-Host "Current Directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found! Creating one..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "Virtual environment created. Installing dependencies..." -ForegroundColor Yellow
    .\.venv\Scripts\pip install -r requirements.txt
}

# Set PYTHONPATH to ensure imports work correctly
$env:PYTHONPATH = $ScriptDir

Write-Host "Starting Uvicorn server..." -ForegroundColor Green
Write-Host "Server will be available at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API Documentation: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Run uvicorn
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
