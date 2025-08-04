#!/usr/bin/env python3
"""
Launcher script for Linear Regression Analyzer
This script provides an easy way to run the application with proper error handling.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'plotly', 'openpyxl', 'xlrd', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    return True

def run_streamlit_app():
    """Run the Streamlit web application."""
    try:
        print("Starting Linear Regression Analyzer...")
        print("The web interface will open in your default browser.")
        print("If it doesn't open automatically, go to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the application.")
        print("-" * 50)
        
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running the application: {e}")
        print("Please make sure all dependencies are installed.")

def run_cli_app():
    """Run the CLI version of the application."""
    try:
        print("Starting Linear Regression Analyzer (CLI version)...")
        print("-" * 50)
        
        # Run CLI app
        subprocess.run([sys.executable, "main.py"])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running the CLI application: {e}")

def main():
    """Main function to handle user choice and run appropriate version."""
    print("=" * 60)
    print("Linear Regression Analyzer - Launcher")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    print("\nChoose how to run the application:")
    print("1. Web Interface (Streamlit) - Recommended")
    print("2. Command Line Interface")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_streamlit_app()
                break
            elif choice == "2":
                run_cli_app()
                break
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 