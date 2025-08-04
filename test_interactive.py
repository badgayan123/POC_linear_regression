#!/usr/bin/env python3
"""
Test script for interactive features
This script tests the interactive functionality without requiring manual input.
"""

import sys
import os
from unittest.mock import patch
from linear_regression_analyzer import LinearRegressionAnalyzer

def test_interactive_features():
    """Test the interactive features of the analyzer"""
    print("Testing Interactive Features...")
    
    # Create analyzer instance
    analyzer = LinearRegressionAnalyzer()
    
    # Test data loading
    print("1. Testing data loading...")
    success = analyzer.load_data('example_data.csv')
    assert success, "Data loading failed"
    print("✓ Data loading successful")
    
    # Test dependent variable selection (simulate user input)
    print("2. Testing dependent variable selection...")
    with patch('builtins.input', return_value='6'):  # Salary is the 6th column
        analyzer.select_dependent_variable()
    assert analyzer.dependent_variable == 'Salary', "Dependent variable not set correctly"
    print("✓ Dependent variable selection successful")
    
    # Test categorical feature identification (simulate user input)
    print("3. Testing categorical feature identification...")
    with patch('builtins.input', return_value='3,4'):  # Education and Department
        analyzer.identify_categorical_features()
    assert 'Education' in analyzer.categorical_features, "Education not identified as categorical"
    assert 'Department' in analyzer.categorical_features, "Department not identified as categorical"
    print("✓ Categorical feature identification successful")
    
    # Test data preprocessing
    print("4. Testing data preprocessing...")
    success = analyzer.preprocess_data()
    assert success, "Data preprocessing failed"
    print("✓ Data preprocessing successful")
    
    # Test model training
    print("5. Testing model training...")
    success = analyzer.train_model()
    assert success, "Model training failed"
    print("✓ Model training successful")
    
    # Test metrics calculation
    print("6. Testing metrics calculation...")
    metrics = analyzer.calculate_metrics()
    assert metrics is not None, "Metrics calculation failed"
    assert 'train_r2' in metrics, "Training R² not calculated"
    assert 'test_r2' in metrics, "Test R² not calculated"
    print("✓ Metrics calculation successful")
    
    # Test prediction
    print("7. Testing prediction...")
    import pandas as pd
    new_data = pd.DataFrame({
        'Age': [30],
        'Experience': [5],
        'Education': ['Bachelor'],
        'Department': ['IT'],
        'Performance_Score': [80]
    })
    result = analyzer.predict_new_data(new_data)
    assert result is not None, "Prediction failed"
    print("✓ Prediction successful")
    
    # Test model saving and loading
    print("8. Testing model persistence...")
    analyzer.save_model('test_model.pkl')
    new_analyzer = LinearRegressionAnalyzer()
    new_analyzer.load_model('test_model.pkl')
    assert new_analyzer.model is not None, "Model loading failed"
    print("✓ Model persistence successful")
    
    # Clean up
    if os.path.exists('test_model.pkl'):
        os.remove('test_model.pkl')
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
    print("The interactive features are working correctly.")

if __name__ == '__main__':
    try:
        test_interactive_features()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 