#!/usr/bin/env python3
"""
Test script to demonstrate the advanced preprocessing features
"""

import pandas as pd
import numpy as np
from linear_regression_analyzer import LinearRegressionAnalyzer

def test_advanced_preprocessing():
    """Test all advanced preprocessing features"""
    print("="*70)
    print("TESTING ADVANCED PREPROCESSING FEATURES")
    print("="*70)
    
    # Create sample data with various characteristics
    np.random.seed(42)
    n_samples = 100
    
    # Generate data with outliers, different scales, and correlations
    data = {
        'Age': np.random.normal(35, 10, n_samples),
        'Experience': np.random.normal(8, 5, n_samples),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], n_samples),
        'Salary': np.random.normal(50000, 15000, n_samples)
    }
    
    # Add some outliers
    data['Age'][0] = 150  # Extreme outlier
    data['Experience'][1] = 50  # Extreme outlier
    data['Salary'][2] = 200000  # Extreme outlier
    
    # Add correlation between features
    data['Salary'] = (data['Age'] * 1000 + 
                     data['Experience'] * 2000 + 
                     np.random.normal(0, 5000, n_samples))
    
    # Add a highly correlated feature (but not Age_Squared to avoid conflicts with polynomial features)
    data['Age_Experience'] = data['Age'] * data['Experience']  # This will be highly correlated
    
    df = pd.DataFrame(data)
    
    print("Sample data created:")
    print(f"Shape: {df.shape}")
    print(f"Salary range: {df['Salary'].min():.2f} to {df['Salary'].max():.2f}")
    print(df.head())
    print()
    
    # Initialize analyzer
    analyzer = LinearRegressionAnalyzer()
    analyzer.data = df
    
    # Set up basic configuration
    analyzer.dependent_variable = 'Salary'
    analyzer.categorical_features = ['Education_Level', 'Department']
    analyzer.numerical_features = ['Age', 'Experience', 'Age_Experience']
    analyzer.dependent_variable_range = 100000
    
    print("Testing different preprocessing configurations...")
    print()
    
    # Test 1: Basic preprocessing (default)
    print("="*50)
    print("TEST 1: BASIC PREPROCESSING (DEFAULT)")
    print("="*50)
    
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
    
    success = analyzer.preprocess_data()
    if success:
        print("Basic preprocessing completed successfully!")
        analyzer.print_preprocessing_summary()
    else:
        print("Basic preprocessing failed!")
    
    print("\n" + "="*50)
    
    # Test 2: Outlier detection and handling
    print("TEST 2: OUTLIER DETECTION & HANDLING")
    print("="*50)
    
    # Reset data
    analyzer.data = df.copy()
    
    analyzer.preprocessing_config = {
        'scaling_method': 'robust',  # Robust to outliers
        'outlier_detection': 'iqr',
        'outlier_handling': 'remove',
        'feature_selection': 'none',
        'feature_selection_params': {},
        'data_transformation': 'none',
        'polynomial_features': False,
        'interaction_features': False,
        'advanced_missing_values': 'auto'
    }
    
    success = analyzer.preprocess_data()
    if success:
        print("Outlier handling completed successfully!")
        analyzer.print_preprocessing_summary()
    else:
        print("Outlier handling failed!")
    
    print("\n" + "="*50)
    
    # Test 3: Feature selection
    print("TEST 3: FEATURE SELECTION")
    print("="*50)
    
    # Reset data
    analyzer.data = df.copy()
    
    analyzer.preprocessing_config = {
        'scaling_method': 'standard',
        'outlier_detection': 'none',
        'outlier_handling': 'remove',
        'feature_selection': 'correlation',
        'feature_selection_params': {'threshold': 0.9},
        'data_transformation': 'none',
        'polynomial_features': False,
        'interaction_features': False,
        'advanced_missing_values': 'auto'
    }
    
    success = analyzer.preprocess_data()
    if success:
        print("Feature selection completed successfully!")
        analyzer.print_preprocessing_summary()
    else:
        print("Feature selection failed!")
    
    print("\n" + "="*50)
    
    # Test 4: Data transformation
    print("TEST 4: DATA TRANSFORMATION")
    print("="*50)
    
    # Reset data
    analyzer.data = df.copy()
    
    analyzer.preprocessing_config = {
        'scaling_method': 'minmax',
        'outlier_detection': 'none',
        'outlier_handling': 'remove',
        'feature_selection': 'none',
        'feature_selection_params': {},
        'data_transformation': 'log',
        'polynomial_features': False,
        'interaction_features': False,
        'advanced_missing_values': 'auto'
    }
    
    success = analyzer.preprocess_data()
    if success:
        print("Data transformation completed successfully!")
        analyzer.print_preprocessing_summary()
    else:
        print("Data transformation failed!")
    
    print("\n" + "="*50)
    
    # Test 5: Feature engineering
    print("TEST 5: FEATURE ENGINEERING")
    print("="*50)
    
    # Reset data
    analyzer.data = df.copy()
    
    analyzer.preprocessing_config = {
        'scaling_method': 'standard',
        'outlier_detection': 'none',
        'outlier_handling': 'remove',
        'feature_selection': 'none',
        'feature_selection_params': {'poly_degree': 2},
        'data_transformation': 'none',
        'polynomial_features': True,
        'interaction_features': True,
        'advanced_missing_values': 'auto'
    }
    
    success = analyzer.preprocess_data()
    if success:
        print("Feature engineering completed successfully!")
        analyzer.print_preprocessing_summary()
    else:
        print("Feature engineering failed!")
    
    print("\n" + "="*50)
    
    # Test 6: Complete pipeline
    print("TEST 6: COMPLETE ADVANCED PIPELINE")
    print("="*50)
    
    # Reset data
    analyzer.data = df.copy()
    
    analyzer.preprocessing_config = {
        'scaling_method': 'robust',
        'outlier_detection': 'zscore',
        'outlier_handling': 'cap',
        'feature_selection': 'variance',
        'feature_selection_params': {'threshold': 0.01},
        'data_transformation': 'boxcox',
        'polynomial_features': False,  # Disable to avoid conflicts
        'interaction_features': False,
        'advanced_missing_values': 'auto'
    }
    
    success = analyzer.preprocess_data()
    if success:
        print("Complete advanced pipeline completed successfully!")
        analyzer.print_preprocessing_summary()
        
        # Train model and test predictions
        print("\nTraining model with advanced preprocessing...")
        if analyzer.train_model():
            print("Model trained successfully!")
            
            # Test predictions
            test_data = pd.DataFrame({
                'Age': [30, 45, 60],
                'Experience': [5, 15, 25],
                'Education_Level': ['Bachelor', 'Master', 'PhD'],
                'Department': ['IT', 'Sales', 'Marketing']
            })
            
            result = analyzer.predict_new_data(test_data)
            if result:
                new_data_with_predictions, predictions = result
                print("\nPredictions with advanced preprocessing:")
                print(new_data_with_predictions)
            else:
                print("Prediction failed!")
        else:
            print("Model training failed!")
    else:
        print("Complete advanced pipeline failed!")
    
    print("\n" + "="*70)
    print("ADVANCED PREPROCESSING TESTING COMPLETED")
    print("="*70)

if __name__ == "__main__":
    test_advanced_preprocessing() 