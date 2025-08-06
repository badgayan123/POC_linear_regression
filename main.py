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

def configure_preprocessing_options(analyzer):
    """Configure advanced preprocessing options"""
    print("\nAdvanced Preprocessing Configuration")
    print("="*40)
    
    while True:
        print("\nPreprocessing Options:")
        print("1. Scaling Method")
        print("2. Outlier Detection & Handling")
        print("3. Feature Selection")
        print("4. Data Transformation")
        print("5. Feature Engineering")
        print("6. Skip advanced preprocessing (use defaults)")
        print("7. Continue with current settings")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            configure_scaling(analyzer)
        elif choice == '2':
            configure_outliers(analyzer)
        elif choice == '3':
            configure_feature_selection(analyzer)
        elif choice == '4':
            configure_transformation(analyzer)
        elif choice == '5':
            configure_feature_engineering(analyzer)
        elif choice == '6':
            # Reset to defaults
            analyzer.preprocessing_config = {
                'scaling_method': 'standard',
                'outlier_detection': 'none',
                'outlier_handling': 'remove',
                'feature_selection': 'none',
                'feature_selection_params': {},
                'data_transformation': 'none',
                'polynomial_features': False,
                'interaction_features': False,
                'advanced_missing_values': 'auto'
            }
            print("Reset to default preprocessing settings.")
            break
        elif choice == '7':
            break
        else:
            print("Invalid choice. Please try again.")

def configure_scaling(analyzer):
    """Configure scaling method"""
    print("\nScaling Methods:")
    print("1. StandardScaler (mean=0, std=1)")
    print("2. MinMaxScaler (0 to 1)")
    print("3. RobustScaler (robust to outliers)")
    print("4. Normalizer (unit norm)")
    print("5. No scaling")
    
    choice = input("\nSelect scaling method (1-5): ").strip()
    
    scaling_methods = {
        '1': 'standard',
        '2': 'minmax',
        '3': 'robust',
        '4': 'normalizer',
        '5': 'none'
    }
    
    if choice in scaling_methods:
        analyzer.preprocessing_config['scaling_method'] = scaling_methods[choice]
        print(f"Scaling method set to: {scaling_methods[choice]}")
    else:
        print("Invalid choice. Using StandardScaler.")

def configure_outliers(analyzer):
    """Configure outlier detection and handling"""
    print("\nOutlier Detection Methods:")
    print("1. IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)")
    print("2. Z-score method (|z| > 3)")
    print("3. Isolation Forest")
    print("4. No outlier detection")
    
    choice = input("\nSelect detection method (1-4): ").strip()
    
    detection_methods = {
        '1': 'iqr',
        '2': 'zscore',
        '3': 'isolation_forest',
        '4': 'none'
    }
    
    if choice in detection_methods:
        analyzer.preprocessing_config['outlier_detection'] = detection_methods[choice]
        
        if detection_methods[choice] != 'none':
            print("\nOutlier Handling Methods:")
            print("1. Remove outliers")
            print("2. Cap outliers (winsorize)")
            print("3. Transform outliers (log transform)")
            
            handle_choice = input("\nSelect handling method (1-3): ").strip()
            
            handling_methods = {
                '1': 'remove',
                '2': 'cap',
                '3': 'transform'
            }
            
            if handle_choice in handling_methods:
                analyzer.preprocessing_config['outlier_handling'] = handling_methods[handle_choice]
                print(f"Outlier detection: {detection_methods[choice]}, handling: {handling_methods[handle_choice]}")
            else:
                print("Invalid choice. Using remove method.")
        else:
            print("Outlier detection disabled.")
    else:
        print("Invalid choice. Using IQR method.")

def configure_feature_selection(analyzer):
    """Configure feature selection"""
    print("\nFeature Selection Methods:")
    print("1. Variance threshold (remove low variance)")
    print("2. Correlation-based (remove highly correlated)")
    print("3. K-best features (F-statistic)")
    print("4. Recursive Feature Elimination (RFE)")
    print("5. No feature selection")
    
    choice = input("\nSelect feature selection method (1-5): ").strip()
    
    selection_methods = {
        '1': 'variance',
        '2': 'correlation',
        '3': 'kbest',
        '4': 'rfe',
        '5': 'none'
    }
    
    if choice in selection_methods:
        method = selection_methods[choice]
        analyzer.preprocessing_config['feature_selection'] = method
        
        if method == 'variance':
            threshold = input("Enter variance threshold (default: 0.01): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.01
                analyzer.preprocessing_config['feature_selection_params'] = {'threshold': threshold}
                print(f"Variance threshold set to: {threshold}")
            except ValueError:
                print("Invalid threshold. Using default 0.01.")
                
        elif method == 'correlation':
            threshold = input("Enter correlation threshold (default: 0.95): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.95
                analyzer.preprocessing_config['feature_selection_params'] = {'threshold': threshold}
                print(f"Correlation threshold set to: {threshold}")
            except ValueError:
                print("Invalid threshold. Using default 0.95.")
                
        elif method == 'kbest':
            k = input("Enter number of features to select (default: 10): ").strip()
            try:
                k = int(k) if k else 10
                analyzer.preprocessing_config['feature_selection_params'] = {'k': k}
                print(f"K-best features set to: {k}")
            except ValueError:
                print("Invalid k. Using default 10.")
                
        elif method == 'rfe':
            n_features = input("Enter number of features to select (default: 10): ").strip()
            try:
                n_features = int(n_features) if n_features else 10
                analyzer.preprocessing_config['feature_selection_params'] = {'n_features': n_features}
                print(f"RFE features set to: {n_features}")
            except ValueError:
                print("Invalid number. Using default 10.")
        else:
            print("Feature selection disabled.")
    else:
        print("Invalid choice. Using no feature selection.")

def configure_transformation(analyzer):
    """Configure data transformation"""
    print("\nData Transformation Methods:")
    print("1. Log transformation (log1p)")
    print("2. Box-Cox transformation")
    print("3. No transformation")
    
    choice = input("\nSelect transformation method (1-3): ").strip()
    
    transformation_methods = {
        '1': 'log',
        '2': 'boxcox',
        '3': 'none'
    }
    
    if choice in transformation_methods:
        analyzer.preprocessing_config['data_transformation'] = transformation_methods[choice]
        print(f"Data transformation set to: {transformation_methods[choice]}")
    else:
        print("Invalid choice. Using no transformation.")

def configure_feature_engineering(analyzer):
    """Configure feature engineering"""
    print("\nFeature Engineering Options:")
    
    # Polynomial features
    poly_choice = input("Add polynomial features? (y/n, default: n): ").strip().lower()
    if poly_choice == 'y':
        degree = input("Enter polynomial degree (default: 2): ").strip()
        try:
            degree = int(degree) if degree else 2
            analyzer.preprocessing_config['polynomial_features'] = True
            analyzer.preprocessing_config['feature_selection_params']['poly_degree'] = degree
            print(f"Polynomial features enabled (degree: {degree})")
        except ValueError:
            print("Invalid degree. Using default 2.")
    else:
        analyzer.preprocessing_config['polynomial_features'] = False
        print("Polynomial features disabled.")
    
    # Interaction features
    interaction_choice = input("Add interaction features? (y/n, default: n): ").strip().lower()
    if interaction_choice == 'y':
        analyzer.preprocessing_config['interaction_features'] = True
        print("Interaction features enabled.")
    else:
        analyzer.preprocessing_config['interaction_features'] = False
        print("Interaction features disabled.")

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
    
    # Step 3: Set dependent variable range
    print("\nStep 3: Set Dependent Variable Range")
    analyzer.set_dependent_variable_range()
    
    # Step 4: Configure advanced preprocessing
    print("\nStep 4: Configure Advanced Preprocessing")
    configure_preprocessing_options(analyzer)
    
    # Step 5: Preprocess data
    print("\nStep 5: Preprocess Data")
    if not analyzer.preprocess_data():
        print("Data preprocessing failed. Please check your data.")
        return
    
    # Step 6: Train model
    print("\nStep 6: Train Model")
    if not analyzer.train_model():
        print("Model training failed.")
        return
    
    # Step 7: Calculate and display metrics
    print("\nStep 7: Calculate Metrics")
    metrics = analyzer.calculate_metrics()
    if metrics:
        analyzer.print_metrics(metrics)
    
    # Step 8: Visualization and additional options
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
