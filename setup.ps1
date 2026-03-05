# setup.ps1 - PowerShell setup script for multilingual summarizer

Write-Host "🚀 Setting up Multilingual PDF Summarizer" -ForegroundColor Green
Write-Host "========================================"

# Check Python installation
Write-Host "`n📋 Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion"

# Check CUDA availability
Write-Host "`n🎮 Checking CUDA availability..." -ForegroundColor Yellow
try {
    $cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaCheck -eq "True") {
        Write-Host "✅ CUDA available - GPU acceleration enabled" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CUDA not available - using CPU (slower)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not check CUDA (PyTorch might not be installed yet)" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host "`n🔧 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Virtual environment created"

# Activate virtual environment
Write-Host "`n🔄 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Virtual environment activated"

# Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed"

# Create necessary directories
Write-Host "`n📁 Creating project directories..." -ForegroundColor Yellow
$directories = @(
    "models",
    "logs",
    "temp",
    "uploads"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir"
    }
}
Write-Host "✅ Directories created"

# Download NLTK data
Write-Host "`n📚 Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" 2>$null
Write-Host "✅ NLTK data downloaded"

# Create .env file if not exists
if (-not (Test-Path ".env")) {
    Write-Host "`n🔐 Creating .env file from template..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" -Destination ".env"
        Write-Host "✅ .env file created. Please update with your credentials."
    } else {
        Write-Host "⚠️  .env.example not found, skipping .env creation" -ForegroundColor Yellow
    }
}

# Run tests
Write-Host "`n🧪 Running tests..." -ForegroundColor Yellow
if (Test-Path "tests") {
    pytest tests/ -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Some tests failed. Check the output above." -ForegroundColor Yellow
    } else {
        Write-Host "✅ All tests passed!" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  Tests directory not found, skipping tests" -ForegroundColor Yellow
}

# Final instructions
Write-Host "`n✨ Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Update the .env file with your credentials (if created)"
Write-Host "  2. Run the API: uvicorn app.main:app --reload"
Write-Host "  3. Or run with Docker: docker-compose up"
Write-Host "  4. Access the API at: http://localhost:8000/docs"
Write-Host "  5. Or use the run script: .\run.ps1 -Mode dev"

Write-Host "`nHappy coding! 🎉" -ForegroundColor Magenta