# Fashion Recommendation Backend - Startup Script

Write-Host "========================================"
Write-Host "Fashion Recommendation Backend Setup"
Write-Host "========================================"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt
Write-Host ""

# Start the server
Write-Host "========================================"
Write-Host "Starting FastAPI server..."
Write-Host "========================================"
Write-Host ""
Write-Host "API will be available at: http://localhost:8000"
Write-Host "API Documentation: http://localhost:8000/docs"
Write-Host ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
