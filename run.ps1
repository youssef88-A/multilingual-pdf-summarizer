param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "prod", "test", "docker", "docker-prod")]
    [string]$Mode = "dev"
)

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Cyan "🚀 Running Multilingual Summarizer in $Mode mode"
Write-ColorOutput Cyan "==========================================="

switch ($Mode) {
    "dev" {
        Write-ColorOutput Yellow "Starting development server..."
        
        # Activate virtual environment if exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & .\venv\Scripts\Activate.ps1
        }
        
        # Run uvicorn with reload
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level info
    }
    
    "prod" {
        Write-ColorOutput Yellow "Starting production server..."
        
        # Activate virtual environment if exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & .\venv\Scripts\Activate.ps1
        }
        
        # Run with gunicorn
        gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300
    }
    
    "test" {
        Write-ColorOutput Yellow "Running tests..."
        
        # Activate virtual environment if exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & .\venv\Scripts\Activate.ps1
        }
        
        # Run pytest with coverage
        pytest tests/ -v --cov=app --cov-report=term --cov-report=html
        
        Write-ColorOutput Green "`nTest coverage report generated in htmlcov/"
    }
    
    "docker" {
        Write-ColorOutput Yellow "Starting Docker containers (development)..."
        
        # Build and run with docker-compose
        docker-compose up --build
    }
    
    "docker-prod" {
        Write-ColorOutput Yellow "Starting Docker containers (production)..."
        
        # Check if .env file exists
        if (-not (Test-Path ".env")) {
            Write-ColorOutput Red "❌ .env file not found. Please create it from .env.example"
            exit 1
        }
        
        # Load environment variables
        Get-Content .env | ForEach-Object {
            if ($_ -match "^(.*?)=(.*)$") {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            }
        }
        
        # Build and run with production docker-compose
        docker-compose -f docker-compose.prod.yml up --build -d
        
        Write-ColorOutput Green "✅ Production containers started in background"
        Write-ColorOutput Yellow "Check logs with: docker-compose -f docker-compose.prod.yml logs -f"
    }
}

# Helper function to stop containers
function Stop-Docker {
    Write-ColorOutput Yellow "Stopping Docker containers..."
    docker-compose down
    docker-compose -f docker-compose.prod.yml down
}

# Helper function to clean up
function Clean-Up {
    Write-ColorOutput Yellow "Cleaning up..."
    
    # Remove Python cache
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force
    
    # Remove pytest cache
    if (Test-Path ".pytest_cache") {
        Remove-Item -Path ".pytest_cache" -Recurse -Force
    }
    
    # Remove coverage reports
    if (Test-Path "htmlcov") {
        Remove-Item -Path "htmlcov" -Recurse -Force
    }
    if (Test-Path ".coverage") {
        Remove-Item -Path ".coverage" -Force
    }
    
    Write-ColorOutput Green "✅ Clean up complete"
}

# Export functions
Export-ModuleMember -Function Stop-Docker, Clean-Up