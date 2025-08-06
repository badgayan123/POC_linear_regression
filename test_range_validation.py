#!/usr/bin/env python3
"""
Test script to demonstrate the new range validation feature
"""

import pandas as pd
import numpy as np
from linear_regression_analyzer import LinearRegressionAnalyzer

def test_range_validation():
    """Test the range validation feature"""
    print("="*60)
    print("TESTING RANGE VALIDATION FEATURE")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'Age': np.random.normal(35, 10, n_samples),
        'Experience': np.random.normal(8, 5, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master'], n_samples),
        'Salary': np.random.normal(50000, 15000, n_samples)
    }
    
    # Add some correlation between features and salary
    data['Salary'] = (data['Age'] * 1000 + 
                     data['Experience'] * 2000 + 
                     np.random.normal(0, 5000, n_samples))
    
    df = pd.DataFrame(data)
    
    # Initialize analyzer
    analyzer = LinearRegressionAnalyzer()
    analyzer.data = df
    
    print("Sample data created:")
    print(f"Salary range: {df['Salary'].min():.2f} to {df['Salary'].max():.2f}")
    print(df.head())
    print()
    
    # Set up the model
    analyzer.dependent_variable = 'Salary'
    analyzer.categorical_features = ['Education']
    analyzer.numerical_features = ['Age', 'Experience']
    
    # Set a conservative maximum range (lower than some actual values)
    analyzer.dependent_variable_range = 70000  # Cap at 70,000
    print(f"Set maximum salary range to: ${analyzer.dependent_variable_range:,.2f}")
    print()
    
    # Preprocess and train
    print("Preprocessing data...")
    analyzer.preprocess_data()
    
    print("Training model...")
    analyzer.train_model()
    
    # Test predictions with extreme values
    print("\n" + "="*50)
    print("TESTING PREDICTIONS WITH EXTREME VALUES")
    print("="*50)
    
    # Create test data with extreme values
    test_data = pd.DataFrame({
        'Age': [25, 80, 45, 100],  # Normal, high, normal, extreme
        'Experience': [5, 30, 15, 50],  # Normal, high, normal, extreme
        'Education': ['Bachelor', 'Master', 'High School', 'Bachelor']
    })
    
    print("Test data with extreme values:")
    print(test_data)
    print()
    
    # Make predictions
    print("Making predictions...")
    result = analyzer.predict_new_data(test_data)
    
    if result:
        new_data_with_predictions, predictions = result
        print("\nPrediction Results:")
        print("="*40)
        
        for i, (_, row) in enumerate(new_data_with_predictions.iterrows()):
            original_pred = predictions[i]
            capped_pred = row['Salary_predicted']
            
            print(f"Sample {i+1}:")
            print(f"  Age: {row['Age']}, Experience: {row['Experience']}, Education: {row['Education']}")
            print(f"  Original prediction: ${original_pred:,.2f}")
            print(f"  Final prediction: ${capped_pred:,.2f}")
            
            if original_pred > analyzer.dependent_variable_range:
                print(f"  ⚠️  CAPPED: Exceeded maximum range of ${analyzer.dependent_variable_range:,.2f}")
            else:
                print(f"  ✅ Within range")
            print()
    
    print("="*60)
    print("RANGE VALIDATION TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_range_validation() 