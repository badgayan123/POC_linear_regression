@echo off
echo Installing Linear Regression Analyzer Dependencies...
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Installation completed successfully!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the web app: python -m streamlit run app.py
echo 3. Or run the CLI version: python main.py
echo.
pause 