# Linear Regression Analysis Tool

A comprehensive Python tool for performing linear regression analysis with Excel and CSV files. This tool provides a complete workflow from data loading to model evaluation and prediction.

## Features

### 📊 Data Handling
- **Multiple File Formats**: Support for Excel (.xlsx, .xls) and CSV files
- **Interactive Variable Selection**: Choose dependent variable with case-insensitive input
- **Categorical Feature Detection**: Automatic one-hot encoding for categorical variables
- **Advanced Data Preprocessing**: Comprehensive preprocessing pipeline with multiple options

### 🔧 Advanced Preprocessing Features
- **Multiple Scaling Methods**: StandardScaler, MinMaxScaler, RobustScaler, Normalizer, or no scaling
- **Outlier Detection & Handling**: IQR, Z-score, and Isolation Forest methods with remove/cap/transform options
- **Feature Selection**: Variance threshold, correlation-based, K-best features, and Recursive Feature Elimination (RFE)
- **Data Transformations**: Log transformation and Box-Cox transformation for skewed data
- **Feature Engineering**: Polynomial features and interaction terms
- **Missing Value Handling**: Advanced imputation strategies including KNN and auto-detection

### 📈 Comprehensive Metrics
- **R² Score**: Coefficient of determination
- **Adjusted R² Score**: R² adjusted for number of features
- **RMSE**: Root Mean Square Error
- **MSE**: Mean Square Error
- **MAE**: Mean Absolute Error
- **Model Coefficients**: Feature importance analysis

### 📊 Visualization
- **Correlation Matrix**: Heatmap showing feature correlations
- **Regression Results**: Actual vs Predicted plots
- **Residual Analysis**: Residual plots for model diagnostics
- **Feature Importance**: Coefficient-based feature importance visualization

### 🔮 Prediction Capabilities
- **New Data Prediction**: Load new files or enter data manually
- **Model Persistence**: Save and load trained models
- **Batch Prediction**: Predict multiple samples at once
- **Range Validation**: Set maximum realistic values to prevent unrealistic predictions

## Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## Usage Guide

### 1. Starting the Application
Run `python main.py` to start the interactive application.

### 2. Loading Data
You have three options:
- **Load your own data file**: Provide path to Excel or CSV file
- **Create sample data**: Generate demonstration data for testing
- **Load saved model**: Use a previously trained model

### 3. Analysis Workflow

#### Step 1: Select Dependent Variable
The tool will display all columns in your dataset. Enter the number corresponding to your target variable.

#### Step 2: Identify Categorical Features
Select which features are categorical (non-numeric). These will be automatically one-hot encoded.

#### Step 3: Set Dependent Variable Range
Set a maximum realistic value for your dependent variable. This prevents the model from making unrealistic predictions when given extreme input values. Predictions exceeding this limit will be automatically capped.

#### Step 4: Configure Advanced Preprocessing
Configure advanced preprocessing options:
- **Scaling**: Choose from StandardScaler, MinMaxScaler, RobustScaler, Normalizer, or no scaling
- **Outlier Detection**: Select IQR, Z-score, or Isolation Forest methods
- **Outlier Handling**: Choose to remove, cap, or transform outliers
- **Feature Selection**: Apply variance threshold, correlation-based, K-best, or RFE selection
- **Data Transformations**: Apply log or Box-Cox transformations
- **Feature Engineering**: Add polynomial features and interaction terms

#### Step 5: Data Preprocessing
The tool automatically:
- Applies configured preprocessing steps
- Scales numerical features according to selection
- Applies one-hot encoding to categorical features
- Splits data into training (80%) and test (20%) sets

#### Step 6: Model Training
Linear regression model is trained on the preprocessed data.

#### Step 7: Metrics Calculation
Comprehensive evaluation metrics are calculated and displayed.

### 4. Visualization and Analysis

After training, you can:
- **View correlation matrix**: Understand feature relationships
- **Plot regression results**: Actual vs predicted values
- **Analyze feature importance**: See which features most influence predictions
- **Make predictions**: Use the trained model on new data
- **Save model**: Store the trained model for later use

### 5. Making Predictions

#### Option A: File-based Prediction
1. Prepare a new data file with the same features (excluding the dependent variable)
2. Load the file through the prediction menu
3. View and optionally save predictions

#### Option B: Manual Data Entry
1. Enter values for each feature manually
2. Get instant prediction results

## Example Workflow

```
1. Start application: python main.py
2. Choose option 2 (Create sample data)
3. Select dependent variable: 5 (Salary)
4. Select categorical features: 3,4 (Education, Department)
5. View metrics and visualizations
6. Make predictions on new data
```

## File Structure

```
pythonProject6_Linear Regression/
├── main.py                          # Main application interface
├── linear_regression_analyzer.py    # Core analysis class
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── sample_data.csv                  # Generated sample data
```

## Data Format Requirements

### Input Data
- **File formats**: .xlsx, .xls, .csv
- **Missing values**: Handled automatically
- **Data types**: Mixed numerical and categorical supported
- **Column names**: Should be descriptive and unique

### For Prediction
- **Same features**: New data must have the same features as training data
- **Same data types**: Categorical features should have the same categories
- **No dependent variable**: Target column should be excluded

## Key Features Explained

### One-Hot Encoding
Categorical variables are automatically converted to binary columns:
- Original: `Education = ['High School', 'Bachelor', 'Master']`
- Encoded: `Education_Bachelor`, `Education_Master` (High School as reference)

### Standardization
Numerical features are scaled to have mean=0 and standard deviation=1:
- Improves model convergence
- Makes coefficients comparable

### Model Evaluation
- **Training metrics**: Performance on training data
- **Test metrics**: Performance on unseen data
- **Cross-validation**: Built-in train/test split

## Troubleshooting

### Common Issues

1. **File not found**: Check file path and ensure file exists
2. **Unsupported format**: Use .xlsx, .xls, or .csv files only
3. **Memory error**: Reduce dataset size or use smaller sample
4. **Import errors**: Install all requirements with `pip install -r requirements.txt`

### Data Quality Tips

1. **Clean your data**: Remove or handle missing values
2. **Check data types**: Ensure categorical variables are strings
3. **Remove duplicates**: Clean dataset before analysis
4. **Feature engineering**: Consider creating new features if needed

## Advanced Usage

### Custom Analysis
You can import the `LinearRegressionAnalyzer` class in your own scripts:

```python
from linear_regression_analyzer import LinearRegressionAnalyzer

analyzer = LinearRegressionAnalyzer()
analyzer.load_data('your_data.csv')
# ... customize analysis as needed
```

### Model Persistence
Save trained models for later use:
```python
analyzer.save_model('my_model.pkl')
analyzer.load_model('my_model.pkl')
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plotting
- **openpyxl**: Excel file reading
- **xlrd**: Legacy Excel file support

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.

---

**Happy Analyzing! 📊🔍** 