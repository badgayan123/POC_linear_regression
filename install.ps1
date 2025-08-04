Write-Host "Installing Linear Regression Analyzer Dependencies..." -ForegroundColor Green
Write-Host ""

Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run the web app: python -m streamlit run app.py" -ForegroundColor White
Write-Host "3. Or run the CLI version: python main.py" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue" 