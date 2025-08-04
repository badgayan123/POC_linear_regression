# PyCharm Setup Guide for Linear Regression Analyzer

## Quick Setup Instructions

### 1. Open Project in PyCharm
1. Open PyCharm
2. Go to `File` → `Open`
3. Navigate to your project folder: `pythonProject6_Linear Regression`
4. Click `OK`

### 2. Configure Python Interpreter
1. Go to `File` → `Settings` (or `PyCharm` → `Preferences` on Mac)
2. Navigate to `Project: pythonProject6_Linear Regression` → `Python Interpreter`
3. Click the gear icon → `Add`
4. Choose `New environment using Virtualenv`
5. Set Python version to 3.8 or higher
6. Click `OK`

### 3. Install Dependencies
1. Open the Terminal in PyCharm (`View` → `Tool Windows` → `Terminal`)
2. Run the installation script:
   ```bash
   # For Windows (PowerShell)
   .\install.ps1
   
   # For Windows (Command Prompt)
   install.bat
   
   # Or manually:
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 4. Configure Run Configurations

#### Streamlit Web App Configuration:
1. Go to `Run` → `Edit Configurations`
2. Click `+` → `Python`
3. Set the following:
   - **Name**: `Streamlit App`
   - **Script path**: Leave empty
   - **Module name**: `streamlit`
   - **Parameters**: `run app.py`
   - **Working directory**: Your project folder
   - **Python interpreter**: Select your virtual environment

#### CLI App Configuration:
1. Click `+` → `Python`
2. Set the following:
   - **Name**: `CLI App`
   - **Script path**: `main.py`
   - **Working directory**: Your project folder
   - **Python interpreter**: Select your virtual environment

### 5. Project Structure
```
pythonProject6_Linear Regression/
├── app.py                          # Streamlit web application
├── main.py                         # CLI version
├── linear_regression_analyzer.py   # Core analysis engine
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── install.bat                     # Windows batch installer
├── install.ps1                     # PowerShell installer
├── README.md                       # Project documentation
├── example_usage.py                # Usage examples
├── test_interactive.py             # Interactive tests
├── example_data.csv                # Sample dataset
└── pycharm_setup.md               # This file
```

### 6. Running the Application

#### Web Interface (Recommended):
1. Select the `Streamlit App` run configuration
2. Click the green play button or press `Shift + F10`
3. The app will open in your default browser at `http://localhost:8501`

#### Command Line Interface:
1. Select the `CLI App` run configuration
2. Click the green play button or press `Shift + F10`

### 7. Debugging
- Set breakpoints by clicking in the left margin of the code editor
- Use `F9` to toggle breakpoints
- Use `F8` to step over, `F7` to step into
- The debugger will work with both Streamlit and CLI versions

### 8. Code Quality Tools
PyCharm will automatically provide:
- Syntax highlighting
- Code completion
- Error detection
- Refactoring tools
- Git integration (if using version control)

### 9. Troubleshooting

#### If Streamlit doesn't start:
1. Check that all dependencies are installed: `pip list`
2. Verify the virtual environment is activated
3. Try running manually: `python -m streamlit run app.py`

#### If you get import errors:
1. Make sure the Python interpreter is set to your virtual environment
2. Reinstall dependencies: `pip install -r requirements.txt`

#### If the web app doesn't open automatically:
1. Check the terminal output for the URL
2. Manually open `http://localhost:8501` in your browser

### 10. Additional PyCharm Features
- **Code Inspection**: PyCharm will highlight potential issues
- **Quick Fix**: Press `Alt + Enter` for quick fixes
- **Refactoring**: Right-click → `Refactor` for code improvements
- **Terminal**: Use the integrated terminal for command-line operations
- **Git Integration**: Built-in Git support for version control

## Support
If you encounter any issues:
1. Check the `README.md` file for detailed documentation
2. Review the `example_usage.py` file for usage examples
3. Ensure all dependencies are properly installed 