import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.dependent_variable = None
        
    def load_data(self, file_path):
        """Load data from Excel or CSV file"""
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Please use .xlsx, .xls, or .csv files.")
            
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            print("\nFirst few rows:")
            print(self.data.head())
            print("\nData types:")
            print(self.data.dtypes)
            print("\nMissing values:")
            print(self.data.isnull().sum())
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def select_dependent_variable(self):
        """Ask user to select dependent variable"""
        print("\nAvailable columns:")
        for i, col in enumerate(self.data.columns, 1):
            print(f"{i}. {col}")
        
        while True:
            try:
                choice = input("\nEnter the number of the dependent variable: ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(self.data.columns):
                    self.dependent_variable = self.data.columns[choice_idx]
                    print(f"Selected dependent variable: {self.dependent_variable}")
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def identify_categorical_features(self):
        """Ask user to identify categorical features"""
        print("\nAvailable columns (excluding dependent variable):")
        available_cols = [col for col in self.data.columns if col != self.dependent_variable]
        
        for i, col in enumerate(available_cols, 1):
            print(f"{i}. {col}")
        
        print("\nEnter the numbers of categorical features (comma-separated, or press Enter if none):")
        choice = input().strip()
        
        if choice:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                self.categorical_features = [available_cols[i] for i in indices if 0 <= i < len(available_cols)]
                print(f"Selected categorical features: {self.categorical_features}")
            except ValueError:
                print("Invalid input. No categorical features selected.")
        
        # Identify numerical features
        self.numerical_features = [col for col in available_cols if col not in self.categorical_features]
        print(f"Numerical features: {self.numerical_features}")
    
    def handle_missing_values(self, method='auto'):
        """
        Handle missing values in the dataset with enhanced reporting
        
        Parameters:
        - method: 'auto', 'mean', 'median', 'mode', 'drop', 'interpolate', 'knn'
        """
        if self.data is None:
            print("❌ No data loaded!")
            return False
        
        # Enhanced missing value analysis
        missing_counts = self.data.isnull().sum()
        total_missing = missing_counts.sum()
        total_cells = len(self.data) * len(self.data.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        if total_missing == 0:
            print("No missing values found in the dataset!")
            return True
        
        # Enhanced reporting
        print(f"\nMISSING VALUE ANALYSIS REPORT")
        print(f"{'='*50}")
        print(f"Total missing values: {total_missing:,}")
        print(f"Missing percentage: {missing_percentage:.2f}%")
        print(f"Total data cells: {total_cells:,}")
        print(f"Affected columns: {len(missing_counts[missing_counts > 0])}")
        print(f"{'='*50}")
        
        print(f"\nDETAILED BREAKDOWN:")
        for col, count in missing_counts[missing_counts > 0].items():
            percentage = (count / len(self.data)) * 100
            severity = "CRITICAL" if percentage > 20 else "MODERATE" if percentage > 5 else "LOW"
            print(f"  {col}: {count:,} ({percentage:.1f}%) - {severity}")
        
        # Auto-detect method with enhanced logic
        if method == 'auto':
            print(f"\nAI AUTO-DETECTION:")
            method = self._auto_detect_missing_value_method()
            print(f"   Selected method: {method.upper()}")
        
        print(f"\nAPPLYING METHOD: {method.upper()}")
        print(f"{'='*50}")
        
        # Apply the selected method with enhanced feedback
        start_time = pd.Timestamp.now()
        
        if method == 'drop':
            self._drop_missing_values()
        elif method == 'mean':
            self._fill_missing_with_mean()
        elif method == 'median':
            self._fill_missing_with_median()
        elif method == 'mode':
            self._fill_missing_with_mode()
        elif method == 'interpolate':
            self._interpolate_missing_values()
        elif method == 'knn':
            self._fill_missing_with_knn()
        else:
            print(f"Unknown method: {method}")
            return False
        
        # Enhanced verification and reporting
        remaining_missing = self.data.isnull().sum().sum()
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        print(f"\nPROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Missing values handled: {total_missing:,}")
        print(f"Remaining missing: {remaining_missing:,}")
        
        if remaining_missing == 0:
            print(f"🎉 SUCCESS: All missing values have been successfully handled!")
            print(f"📈 Data quality improved by {missing_percentage:.2f}%")
        else:
            print(f"⚠️ WARNING: {remaining_missing} missing values remain after handling.")
            print(f"💡 Consider using a different method or manual inspection.")
        
        print(f"{'='*50}")
        return True
    
    def _auto_detect_missing_value_method(self):
        """Auto-detect the best method for handling missing values with detailed reasoning"""
        missing_counts = self.data.isnull().sum()
        total_rows = len(self.data)
        total_missing = missing_counts.sum()
        
        print(f"   🤖 AI ANALYSIS:")
        print(f"      Total missing: {total_missing:,}")
        print(f"      Total rows: {total_rows:,}")
        print(f"      Missing percentage: {(total_missing/total_rows)*100:.1f}%")
        
        # If more than 50% of data is missing, drop rows
        if total_missing > total_rows * 0.5:
            print(f"      ⚠️ High missing data (>50%) → DROP ROWS")
            return 'drop'
        
        # Check each column for extreme missing values
        columns_to_drop = []
        for col in missing_counts[missing_counts > 0].index:
            missing_pct = missing_counts[col] / total_rows
            
            # If more than 30% missing in a column, mark for dropping
            if missing_pct > 0.3:
                print(f"      🗑️ Column '{col}' has {missing_pct*100:.1f}% missing → DROP COLUMN")
                columns_to_drop.append(col)
        
        # Drop problematic columns
        if columns_to_drop:
            print(f"      📊 Dropping {len(columns_to_drop)} columns with >30% missing values")
            self.data = self.data.drop(columns=columns_to_drop)
            
            # Update feature lists
            for col in columns_to_drop:
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
        
        # Analyze remaining data characteristics
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print(f"      📈 Numerical columns: {len(numerical_cols)}")
        print(f"      🏆 Categorical columns: {len(categorical_cols)}")
        
        # Check for outliers in numerical columns
        outlier_analysis = []
        has_outliers = False
        
        for col in numerical_cols:
            if col in self.data.columns and self.data[col].notna().sum() > 0:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outliers / self.data[col].notna().sum()) * 100
                
                if outliers > 0:
                    has_outliers = True
                    outlier_analysis.append(f"{col}: {outliers} outliers ({outlier_pct:.1f}%)")
        
        if outlier_analysis:
            print(f"      🔍 Outlier detection:")
            for analysis in outlier_analysis:
                print(f"         {analysis}")
            print(f"      📊 Outliers detected → MEDIAN (robust to outliers)")
            return 'median'
        else:
            print(f"      ✅ No significant outliers → MEAN (good for normal distributions)")
            return 'mean'
    
    def _drop_missing_values(self):
        """Drop rows with missing values with enhanced reporting"""
        initial_rows = len(self.data)
        initial_missing = self.data.isnull().sum().sum()
        
        print(f"🗑️ DROPPING ROWS WITH MISSING VALUES")
        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Initial missing values: {initial_missing:,}")
        
        self.data = self.data.dropna()
        dropped_rows = initial_rows - len(self.data)
        final_rows = len(self.data)
        
        print(f"   Rows dropped: {dropped_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Data reduction: {(dropped_rows/initial_rows)*100:.1f}%")
        print(f"   ✅ All missing values eliminated!")
    
    def _fill_missing_with_mean(self):
        """Fill missing values with mean for numerical columns with enhanced reporting"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        total_filled = 0
        
        print(f"📊 FILLING MISSING VALUES WITH MEAN")
        print(f"   Numerical columns found: {len(numerical_cols)}")
        
        for col in numerical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                mean_val = self.data[col].mean()
                self.data[col].fillna(mean_val, inplace=True)
                total_filled += missing_count
                print(f"   📈 '{col}': {missing_count:,} values → mean {mean_val:.2f}")
        
        print(f"   ✅ Total values filled: {total_filled:,}")
    
    def _fill_missing_with_median(self):
        """Fill missing values with median for numerical columns with enhanced reporting"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        total_filled = 0
        
        print(f"📈 FILLING MISSING VALUES WITH MEDIAN")
        print(f"   Numerical columns found: {len(numerical_cols)}")
        
        for col in numerical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                total_filled += missing_count
                print(f"   📊 '{col}': {missing_count:,} values → median {median_val:.2f}")
        
        print(f"   ✅ Total values filled: {total_filled:,}")
    
    def _fill_missing_with_mode(self):
        """Fill missing values with mode for categorical columns with enhanced reporting"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        total_filled = 0
        
        print(f"🏆 FILLING MISSING VALUES WITH MODE")
        print(f"   Categorical columns found: {len(categorical_cols)}")
        
        for col in categorical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                mode_val = self.data[col].mode()[0]
                self.data[col].fillna(mode_val, inplace=True)
                total_filled += missing_count
                print(f"   🎯 '{col}': {missing_count:,} values → mode '{mode_val}'")
        
        print(f"   ✅ Total values filled: {total_filled:,}")
    
    def _interpolate_missing_values(self):
        """Interpolate missing values (good for time series data) with enhanced reporting"""
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        total_filled = 0
        
        print(f"📈 INTERPOLATING MISSING VALUES")
        print(f"   Numerical columns found: {len(numerical_cols)}")
        
        for col in numerical_cols:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                self.data[col] = self.data[col].interpolate(method='linear')
                total_filled += missing_count
                print(f"   🔄 '{col}': {missing_count:,} values interpolated")
        
        print(f"   ✅ Total values interpolated: {total_filled:,}")
    
    def _fill_missing_with_knn(self):
        """Fill missing values using K-Nearest Neighbors with enhanced reporting"""
        try:
            from sklearn.impute import KNNImputer
            
            print(f"🧠 FILLING MISSING VALUES WITH KNN IMPUTATION")
            
            # Separate numerical and categorical columns
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            
            total_numerical_filled = 0
            total_categorical_filled = 0
            
            # Handle numerical columns with KNN
            if len(numerical_cols) > 0:
                print(f"   📊 Processing {len(numerical_cols)} numerical columns with KNN...")
                knn_imputer = KNNImputer(n_neighbors=5)
                
                # Count missing values before imputation
                for col in numerical_cols:
                    total_numerical_filled += self.data[col].isnull().sum()
                
                self.data[numerical_cols] = knn_imputer.fit_transform(self.data[numerical_cols])
                print(f"   ✅ KNN imputation completed for numerical columns")
                print(f"   📈 Values filled: {total_numerical_filled:,}")
            
            # Handle categorical columns with mode
            if len(categorical_cols) > 0:
                print(f"   🏆 Processing {len(categorical_cols)} categorical columns with mode...")
                
                for col in categorical_cols:
                    missing_count = self.data[col].isnull().sum()
                    if missing_count > 0:
                        mode_val = self.data[col].mode()[0]
                        self.data[col].fillna(mode_val, inplace=True)
                        total_categorical_filled += missing_count
                        print(f"   🎯 '{col}': {missing_count:,} values → mode '{mode_val}'")
                
                print(f"   ✅ Mode imputation completed for categorical columns")
                print(f"   📊 Values filled: {total_categorical_filled:,}")
            
            total_filled = total_numerical_filled + total_categorical_filled
            print(f"   🎉 TOTAL VALUES FILLED: {total_filled:,}")
                    
        except ImportError:
            print("❌ KNN imputation not available. Falling back to mean imputation.")
            self._fill_missing_with_mean()
    
    def preprocess_data(self, missing_method='auto'):
        """Preprocess data with one-hot encoding for categorical variables"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        # Handle missing values first
        print("🔍 Checking for missing values...")
        self.handle_missing_values(method=missing_method)
        
        # Separate features and target
        X = self.data.drop(columns=[self.dependent_variable])
        y = self.data[self.dependent_variable]
        
        # Create preprocessing pipeline
        preprocessors = []
        
        print(f"Debug: numerical_features = {self.numerical_features}")
        print(f"Debug: categorical_features = {self.categorical_features}")
        
        if self.numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            preprocessors.append(('num', numerical_transformer, self.numerical_features))
        
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            preprocessors.append(('cat', categorical_transformer, self.categorical_features))
        
        # Always create a preprocessor - at minimum we'll have numerical features
        if not preprocessors:
            # If no preprocessors were added, it means no features were identified
            # This shouldn't happen, but let's handle it gracefully
            print("No features identified for preprocessing!")
            return False
            
        self.preprocessor = ColumnTransformer(transformers=preprocessors, remainder='drop')
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit preprocessor and transform data
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Get feature names after preprocessing
        feature_names = []
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        if self.categorical_features:
            # Get one-hot encoded feature names
            cat_encoder = self.preprocessor.named_transformers_['cat']['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
        
        # Store categorical categories for reference
        if self.categorical_features:
            self.categorical_categories = {}
            for cat_feature in self.categorical_features:
                if 'cat' in self.preprocessor.named_transformers_:
                    cat_encoder = self.preprocessor.named_transformers_['cat']['onehot']
                    # Get the index of this categorical feature in the categorical_features list
                    cat_index = self.categorical_features.index(cat_feature)
                    self.categorical_categories[cat_feature] = cat_encoder.categories_[cat_index].tolist()
        
        print(f"Data preprocessed successfully!")
        print(f"Training set shape: {self.X_train_processed.shape}")
        print(f"Test set shape: {self.X_test_processed.shape}")
        
        if self.categorical_features:
            print("\nCategorical feature categories:")
            for feature, categories in self.categorical_categories.items():
                print(f"  {feature}: {categories}")
        
        return True
    
    def train_model(self):
        """Train the linear regression model"""
        if self.X_train_processed is None:
            print("Data not preprocessed. Please preprocess data first.")
            return False
        
        self.model = LinearRegression()
        self.model.fit(self.X_train_processed, self.y_train)
        
        print("Model trained successfully!")
        return True
    
    def calculate_metrics(self):
        """Calculate all regression metrics"""
        if self.model is None:
            print("Model not trained. Please train model first.")
            return None
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_processed)
        y_test_pred = self.model.predict(self.X_test_processed)
        
        # Calculate metrics
        metrics = {}
        
        # Training metrics
        metrics['train_r2'] = r2_score(self.y_train, y_train_pred)
        metrics['train_adj_r2'] = 1 - (1 - metrics['train_r2']) * (len(self.y_train) - 1) / (len(self.y_train) - len(self.feature_names) - 1)
        metrics['train_mse'] = mean_squared_error(self.y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['train_mae'] = mean_absolute_error(self.y_train, y_train_pred)
        
        # Test metrics
        metrics['test_r2'] = r2_score(self.y_test, y_test_pred)
        metrics['test_adj_r2'] = 1 - (1 - metrics['test_r2']) * (len(self.y_test) - 1) / (len(self.y_test) - len(self.feature_names) - 1)
        metrics['test_mse'] = mean_squared_error(self.y_test, y_test_pred)
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        metrics['test_mae'] = mean_absolute_error(self.y_test, y_test_pred)
        
        # Store predictions for plotting
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print all calculated metrics"""
        print("\n" + "="*50)
        print("REGRESSION METRICS")
        print("="*50)
        
        print("\nTRAINING SET METRICS:")
        print(f"R² Score: {metrics['train_r2']:.4f}")
        print(f"Adjusted R² Score: {metrics['train_adj_r2']:.4f}")
        print(f"Mean Squared Error (MSE): {metrics['train_mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['train_rmse']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['train_mae']:.4f}")
        
        print("\nTEST SET METRICS:")
        print(f"R² Score: {metrics['test_r2']:.4f}")
        print(f"Adjusted R² Score: {metrics['test_adj_r2']:.4f}")
        print(f"Mean Squared Error (MSE): {metrics['test_mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['test_rmse']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['test_mae']:.4f}")
        
        print("\nMODEL COEFFICIENTS:")
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
        print(coefficients)
        print(f"\nIntercept: {self.model.intercept_:.4f}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix"""
        plt.figure(figsize=(12, 10))
        
        # Select only numerical columns for correlation
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numerical_data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix (Numerical Variables Only)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_regression_results(self):
        """Plot regression results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted (Training)
        axes[0, 0].scatter(self.y_train, self.y_train_pred, alpha=0.6)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Training: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted (Test)
        axes[0, 1].scatter(self.y_test, self.y_test_pred, alpha=0.6, color='orange')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Test: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals (Training)
        residuals_train = self.y_train - self.y_train_pred
        axes[1, 0].scatter(self.y_train_pred, residuals_train, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Training: Residuals vs Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals (Test)
        residuals_test = self.y_test - self.y_test_pred
        axes[1, 1].scatter(self.y_test_pred, residuals_test, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Test: Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance based on coefficients"""
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in coefficients['Coefficient']]
        plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Linear Regression Coefficients)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_new_data(self, new_data):
        """Predict outcomes for new data"""
        if self.model is None or self.preprocessor is None:
            print("Model not trained. Please train model first.")
            return None
        
        try:
            # Check if all required features are present
            required_features = self.numerical_features + self.categorical_features
            missing_features = [col for col in required_features if col not in new_data.columns]
            if missing_features:
                print(f"Missing features in new data: {missing_features}")
                return None
            
            # Validate categorical values before preprocessing
            if hasattr(self, 'categorical_categories') and self.categorical_categories:
                invalid_categories = {}
                for feature in self.categorical_features:
                    if feature in new_data.columns:
                        unique_values = new_data[feature].unique()
                        valid_categories = self.categorical_categories[feature]
                        invalid_values = [val for val in unique_values if val not in valid_categories]
                        if invalid_values:
                            invalid_categories[feature] = {
                                'invalid_values': invalid_values,
                                'valid_categories': valid_categories
                            }
                
                if invalid_categories:
                    print("Invalid categorical values found:")
                    for feature, info in invalid_categories.items():
                        print(f"  {feature}:")
                        print(f"    Invalid values: {info['invalid_values']}")
                        print(f"    Valid categories: {info['valid_categories']}")
                    print("Please use only the valid categories shown above.")
                    return None
            
            # Preprocess new data
            new_data_processed = self.preprocessor.transform(new_data)
            
            # Make predictions
            predictions = self.model.predict(new_data_processed)
            
            # Add predictions to the dataframe
            new_data_with_predictions = new_data.copy()
            new_data_with_predictions[f'{self.dependent_variable}_predicted'] = predictions
            
            return new_data_with_predictions, predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            if "Found unknown categories" in str(e):
                print("This error occurs when you enter categorical values that don't exist in your training data.")
                print("Please use only the categories that were present in your original dataset.")
                if hasattr(self, 'categorical_categories') and self.categorical_categories:
                    print("Valid categories for each feature:")
                    for feature, categories in self.categorical_categories.items():
                        print(f"  {feature}: {categories}")
            else:
                print("Make sure the new data has the same structure as the training data.")
            return None
    
    def save_model(self, filepath):
        """Save the trained model and preprocessor"""
        import joblib
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'dependent_variable': self.dependent_variable,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessor"""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_names = model_data['feature_names']
        self.dependent_variable = model_data['dependent_variable']
        self.categorical_features = model_data['categorical_features']
        self.numerical_features = model_data['numerical_features']
        print(f"Model loaded from {filepath}") 