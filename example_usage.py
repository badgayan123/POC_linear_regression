#!/usr/bin/env python3
"""
Example usage of the Linear Regression Analysis Tool
This script demonstrates how to use the LinearRegressionAnalyzer class programmatically.
"""

import pandas as pd
import numpy as np
from linear_regression_analyzer import LinearRegressionAnalyzer

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate features
    age = np.random.normal(35, 10, n_samples)
    experience = np.random.normal(8, 5, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    department = np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n_samples)
    performance_score = np.random.normal(75, 15, n_samples)
    
    # Create target variable with some correlation to features
    salary = (age * 800 + 
             experience * 1500 + 
             performance_score * 200 + 
             np.random.normal(0, 3000, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Education': education,
        'Department': department,
        'Performance_Score': performance_score,
        'Salary': salary
    })
    
    return data

def main():
    """Main example function"""
    print("="*60)
    print("LINEAR REGRESSION ANALYSIS - EXAMPLE USAGE")
    print("="*60)
    
    # Create analyzer instance
    analyzer = LinearRegressionAnalyzer()
    
    # Create sample data
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    # Save sample data
    sample_data.to_csv('example_data.csv', index=False)
    print("Sample data saved as 'example_data.csv'")
    
    # Load data into analyzer
    print("\nLoading data into analyzer...")
    analyzer.data = sample_data
    
    print(f"Data loaded! Shape: {analyzer.data.shape}")
    print("\nFirst few rows:")
    print(analyzer.data.head())
    
    # Set dependent variable (Salary)
    analyzer.dependent_variable = 'Salary'
    print(f"\nDependent variable set to: {analyzer.dependent_variable}")
    
    # Set categorical features
    analyzer.categorical_features = ['Education', 'Department']
    analyzer.numerical_features = ['Age', 'Experience', 'Performance_Score']
    print(f"Categorical features: {analyzer.categorical_features}")
    print(f"Numerical features: {analyzer.numerical_features}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    if analyzer.preprocess_data():
        print("Data preprocessing completed successfully!")
    else:
        print("Data preprocessing failed!")
        return
    
    # Train model
    print("\nTraining linear regression model...")
    if analyzer.train_model():
        print("Model training completed successfully!")
    else:
        print("Model training failed!")
        return
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = analyzer.calculate_metrics()
    if metrics:
        analyzer.print_metrics(metrics)
    else:
        print("Failed to calculate metrics!")
        return
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Correlation matrix
    print("1. Correlation matrix...")
    analyzer.plot_correlation_matrix()
    
    # Regression results
    print("2. Regression results plots...")
    analyzer.plot_regression_results()
    
    # Feature importance
    print("3. Feature importance plot...")
    analyzer.plot_feature_importance()
    
    # Make predictions on new data
    print("\nMaking predictions on new data...")
    
    # Create new data for prediction
    new_data = pd.DataFrame({
        'Age': [30, 45, 28],
        'Experience': [5, 15, 3],
        'Education': ['Bachelor', 'Master', 'High School'],
        'Department': ['IT', 'Sales', 'HR'],
        'Performance_Score': [80, 90, 70]
    })
    
    print("New data for prediction:")
    print(new_data)
    
    # Make predictions
    result = analyzer.predict_new_data(new_data)
    if result:
        new_data_with_predictions, predictions = result
        print("\nPredictions:")
        print(new_data_with_predictions)
        
        # Save predictions
        new_data_with_predictions.to_csv('predictions.csv', index=False)
        print("\nPredictions saved to 'predictions.csv'")
    
    # Save model
    print("\nSaving trained model...")
    analyzer.save_model('example_model.pkl')
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files created:")
    print("- example_data.csv: Sample dataset")
    print("- predictions.csv: Prediction results")
    print("- example_model.pkl: Trained model")
    print("\nYou can now run 'python main.py' for interactive analysis!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc() 