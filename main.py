#!/usr/bin/env python3
"""
Linear Regression Analysis Tool
A comprehensive tool for performing linear regression analysis with Excel and CSV files.
"""

import os
import sys
import pandas as pd
from linear_regression_analyzer import LinearRegressionAnalyzer

def print_banner():
    """Print application banner"""
    print("="*60)
    print("           LINEAR REGRESSION ANALYSIS TOOL")
    print("="*60)
    print("Features:")
    print("• Load Excel (.xlsx, .xls) and CSV files")
    print("• Automatic data preprocessing with one-hot encoding")
    print("• Comprehensive regression metrics (R², Adjusted R², RMSE, MSE, MAE)")
    print("• Correlation analysis and visualization")
    print("• Model training and evaluation")
    print("• Prediction on new data")
    print("• Model saving and loading")
    print("="*60)

def get_file_path():
    """Get file path from user"""
    while True:
        file_path = input("\nEnter the path to your data file (.xlsx, .xls, or .csv): ").strip()
        
        # Remove quotes if present
        file_path = file_path.strip('"\'')
        
        if not os.path.exists(file_path):
            print("File not found. Please check the path and try again.")
            continue
        
        if not file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            print("Unsupported file format. Please use .xlsx, .xls, or .csv files.")
            continue
        
        return file_path

def create_sample_data():
    """Create sample data for demonstration"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample data
    data = {
        'Age': np.random.normal(35, 10, n_samples),
        'Experience': np.random.normal(8, 5, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n_samples),
        'Salary': np.random.normal(50000, 15000, n_samples)
    }
    
    # Add some correlation between features and salary
    data['Salary'] = (data['Age'] * 1000 + 
                     data['Experience'] * 2000 + 
                     np.random.normal(0, 5000, n_samples))
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    sample_file = 'sample_data.csv'
    df.to_csv(sample_file, index=False)
    print(f"Sample data created and saved as '{sample_file}'")
    return sample_file

def main():
    """Main application function"""
    print_banner()
    
    analyzer = LinearRegressionAnalyzer()
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Load data from file")
        print("2. Create sample data for testing")
        print("3. Load saved model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Load data
            file_path = get_file_path()
            if analyzer.load_data(file_path):
                run_analysis(analyzer)
            else:
                print("Failed to load data. Please try again.")
        
        elif choice == '2':
            # Create sample data
            sample_file = create_sample_data()
            if analyzer.load_data(sample_file):
                print("\nSample data loaded successfully!")
                run_analysis(analyzer)
            else:
                print("Failed to load sample data.")
        
        elif choice == '3':
            # Load saved model
            model_path = input("Enter the path to the saved model file: ").strip()
            model_path = model_path.strip('"\'')
            if os.path.exists(model_path):
                try:
                    analyzer.load_model(model_path)
                    print("Model loaded successfully!")
                    run_prediction_mode(analyzer)
                except Exception as e:
                    print(f"Error loading model: {e}")
            else:
                print("Model file not found.")
        
        elif choice == '4':
            print("Thank you for using the Linear Regression Analysis Tool!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def run_analysis(analyzer):
    """Run the complete analysis workflow"""
    print("\n" + "="*50)
    print("ANALYSIS WORKFLOW")
    print("="*50)
    
    # Step 1: Select dependent variable
    print("\nStep 1: Select Dependent Variable")
    analyzer.select_dependent_variable()
    
    # Step 2: Identify categorical features
    print("\nStep 2: Identify Categorical Features")
    analyzer.identify_categorical_features()
    
    # Step 3: Preprocess data
    print("\nStep 3: Preprocess Data")
    if not analyzer.preprocess_data():
        print("Data preprocessing failed. Please check your data.")
        return
    
    # Step 4: Train model
    print("\nStep 4: Train Model")
    if not analyzer.train_model():
        print("Model training failed.")
        return
    
    # Step 5: Calculate and display metrics
    print("\nStep 5: Calculate Metrics")
    metrics = analyzer.calculate_metrics()
    if metrics:
        analyzer.print_metrics(metrics)
    
    # Step 6: Visualization and additional options
    run_visualization_menu(analyzer)

def run_visualization_menu(analyzer):
    """Run visualization and additional options menu"""
    while True:
        print("\n" + "="*50)
        print("VISUALIZATION & ANALYSIS MENU")
        print("="*50)
        print("1. Plot correlation matrix")
        print("2. Plot regression results (Actual vs Predicted)")
        print("3. Plot feature importance")
        print("4. Make predictions on new data")
        print("5. Save model")
        print("6. Return to main menu")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("Generating correlation matrix...")
            analyzer.plot_correlation_matrix()
        
        elif choice == '2':
            print("Generating regression results plots...")
            analyzer.plot_regression_results()
        
        elif choice == '3':
            print("Generating feature importance plot...")
            analyzer.plot_feature_importance()
        
        elif choice == '4':
            run_prediction_mode(analyzer)
        
        elif choice == '5':
            model_path = input("Enter path to save model (e.g., model.pkl): ").strip()
            model_path = model_path.strip('"\'')
            analyzer.save_model(model_path)
        
        elif choice == '6':
            break
        
        else:
            print("Invalid choice. Please try again.")

def run_prediction_mode(analyzer):
    """Run prediction mode for new data"""
    print("\n" + "="*50)
    print("PREDICTION MODE")
    print("="*50)
    
    if analyzer.model is None:
        print("No trained model available. Please train a model first.")
        return
    
    print("Options:")
    print("1. Load new data file for prediction")
    print("2. Enter data manually")
    print("3. Return to previous menu")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Load new data file
        file_path = get_file_path()
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                new_data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                new_data = pd.read_csv(file_path)
            
            # Remove dependent variable column if present
            if analyzer.dependent_variable in new_data.columns:
                new_data = new_data.drop(columns=[analyzer.dependent_variable])
            
            result = analyzer.predict_new_data(new_data)
            if result:
                new_data_with_predictions, predictions = result
                print("\nPredictions:")
                print(new_data_with_predictions)
                
                # Save predictions
                save_choice = input("\nSave predictions to file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_file = input("Enter output file name (e.g., predictions.csv): ").strip()
                    new_data_with_predictions.to_csv(output_file, index=False)
                    print(f"Predictions saved to {output_file}")
        
        except Exception as e:
            print(f"Error loading new data: {e}")
    
    elif choice == '2':
        # Manual data entry
        print("\nManual Data Entry")
        print("Enter values for each feature (press Enter to skip):")
        
        # Get feature names from original data
        original_features = [col for col in analyzer.data.columns if col != analyzer.dependent_variable]
        
        manual_data = {}
        for feature in original_features:
            value = input(f"{feature}: ").strip()
            if value:
                # Try to convert to appropriate type
                try:
                    if feature in analyzer.numerical_features:
                        manual_data[feature] = float(value)
                    else:
                        manual_data[feature] = value
                except ValueError:
                    print(f"Invalid value for {feature}. Skipping.")
        
        if manual_data:
            new_data = pd.DataFrame([manual_data])
            result = analyzer.predict_new_data(new_data)
            if result:
                new_data_with_predictions, predictions = result
                print(f"\nPredicted {analyzer.dependent_variable}: {predictions[0]:.2f}")
    
    elif choice == '3':
        return
    
    else:
        print("Invalid choice. Please try again.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
