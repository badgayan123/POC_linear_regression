# Quick Start Guide - Linear Regression Analysis Tool

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Follow the Interactive Menu
- Choose option 2 to create sample data
- Select dependent variable (e.g., Salary)
- Identify categorical features (e.g., Education, Department)
- View results and visualizations

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Interactive application interface |
| `linear_regression_analyzer.py` | Core analysis engine |
| `example_usage.py` | Programmatic usage example |
| `requirements.txt` | Python dependencies |
| `README.md` | Comprehensive documentation |

## 🎯 Key Features

### ✅ What's Included
- **File Support**: Excel (.xlsx, .xls) and CSV files
- **Interactive Selection**: Choose dependent variable and categorical features
- **Automatic Preprocessing**: One-hot encoding + standardization
- **Complete Metrics**: R², Adjusted R², RMSE, MSE, MAE
- **Visualizations**: Correlation matrix, regression plots, feature importance
- **Predictions**: New data prediction with file or manual input
- **Model Persistence**: Save and load trained models

### 📊 Example Output
```
REGRESSION METRICS
==================

TRAINING SET METRICS:
R² Score: 0.9457
Adjusted R² Score: 0.9420
RMSE: 2692.2808
MSE: 7248375.9093
MAE: 2104.1021

TEST SET METRICS:
R² Score: 0.9534
Adjusted R² Score: 0.9373
RMSE: 2639.3849
MSE: 6966352.6419
MAE: 2134.9686
```

## 🔧 Usage Examples

### Interactive Mode
```bash
python main.py
# Follow the menu prompts
```

### Programmatic Mode
```python
from linear_regression_analyzer import LinearRegressionAnalyzer

analyzer = LinearRegressionAnalyzer()
analyzer.load_data('your_data.csv')
analyzer.dependent_variable = 'target_column'
analyzer.categorical_features = ['cat_feature1', 'cat_feature2']
analyzer.preprocess_data()
analyzer.train_model()
metrics = analyzer.calculate_metrics()
```

### Quick Test
```bash
python example_usage.py
```

## 📈 Sample Workflow

1. **Load Data**: Excel or CSV file
2. **Select Target**: Choose dependent variable
3. **Identify Categories**: Mark categorical features
4. **Train Model**: Automatic preprocessing + training
5. **View Results**: Metrics and visualizations
6. **Make Predictions**: New data or manual entry
7. **Save Model**: For future use

## 🛠️ Troubleshooting

### Common Issues
- **File not found**: Check file path and format
- **Import errors**: Run `pip install -r requirements.txt`
- **Memory issues**: Use smaller datasets
- **Categorical errors**: Ensure categorical variables are strings

### Data Requirements
- **Format**: .xlsx, .xls, or .csv
- **Missing values**: Handled automatically
- **Mixed types**: Numerical and categorical supported
- **Column names**: Descriptive and unique

## 🎉 Ready to Analyze!

Your linear regression analysis tool is ready to use. Start with the sample data to see all features in action, then load your own datasets for analysis.

**Happy Analyzing! 📊🔍** 