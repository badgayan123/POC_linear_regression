#!/usr/bin/env python3
"""
Deployment Helper for Streamlit Cloud
This script helps prepare and deploy your Linear Regression Analyzer to Streamlit Cloud.
"""

import os
import sys
import subprocess
import webbrowser

def check_files():
    """Check if all necessary files exist for deployment."""
    required_files = [
        'app.py',
        'linear_regression_analyzer.py',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files for deployment:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("All required files are present!")
    return True

def check_git_status():
    """Check Git status and provide guidance."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("You have uncommitted changes:")
            print(result.stdout)
            print("Please commit your changes before deploying:")
            print("  git add .")
            print("  git commit -m 'Your commit message'")
            print("  git push origin master")
            return False
        else:
            print("Git repository is clean!")
            return True
    except Exception as e:
        print(f"Error checking Git status: {e}")
        return False

def open_streamlit_cloud():
    """Open Streamlit Cloud in the browser."""
    url = "https://share.streamlit.io"
    print(f"Opening Streamlit Cloud: {url}")
    webbrowser.open(url)

def show_deployment_steps():
    """Show step-by-step deployment instructions."""
    print("\n" + "="*60)
    print("STREAMLIT CLOUD DEPLOYMENT STEPS")
    print("="*60)
    
    print("\n1. Go to https://share.streamlit.io")
    print("2. Sign in with your GitHub account")
    print("3. Click 'New app'")
    print("4. Select your repository: pythonProject6_Linear Regression")
    print("5. Set the main file path: app.py")
    print("6. Click 'Deploy!'")
    
    print("\nIMPORTANT NOTES:")
    print("- Make sure your repository is public (for free tier)")
    print("- The app.py file should be in the root directory")
    print("- All dependencies are in requirements.txt")
    print("- Configuration is in .streamlit/config.toml")
    
    print("\nTROUBLESHOOTING:")
    print("- If deployment fails, check the logs in Streamlit Cloud")
    print("- Ensure all imports work correctly")
    print("- Test locally first: streamlit run app.py")

def main():
    """Main deployment helper function."""
    print("="*60)
    print("Linear Regression Analyzer - Deployment Helper")
    print("="*60)
    
    # Check files
    print("\nChecking required files...")
    if not check_files():
        print("\nPlease ensure all required files are present before deploying.")
        return
    
    # Check Git status
    print("\nChecking Git status...")
    if not check_git_status():
        print("\nPlease commit and push your changes before deploying.")
        return
    
    # Show deployment steps
    show_deployment_steps()
    
    # Ask if user wants to open Streamlit Cloud
    print("\n" + "-"*60)
    choice = input("Would you like to open Streamlit Cloud now? (y/n): ").lower()
    
    if choice in ['y', 'yes']:
        open_streamlit_cloud()
    
    print("\nDeployment helper completed!")
    print("Follow the steps above to deploy your app to Streamlit Cloud.")

if __name__ == "__main__":
    main() 