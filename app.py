import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from linear_regression_analyzer import LinearRegressionAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Linear Regression Analysis Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Black theme base styling */
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    /* Metric cards with black theme */
    .metric-card {
        background: #111111;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #333333;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1);
    }
    
    /* Feature cards with black theme */
    .feature-card {
        background: #111111;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #ffffff;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    /* Success box with black theme */
    .success-box {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1);
        border: 1px solid #333333;
    }
    
    /* Info box with black theme */
    .info-box {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255, 255, 255, 0.1);
        border: 1px solid #333333;
    }
    
    /* Sidebar styling with black theme */
    .css-1d391kg {
        background: #000000;
        border-right: 1px solid #333333;
    }
    
    /* Button styling with black theme */
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #f0f0f0;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.3);
    }
    
    /* Text color adjustments for dark theme */
    .stMarkdown, .stText {
        color: #ffffff !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: #111111;
        border: 2px dashed #333333;
        border-radius: 15px;
        color: #ffffff;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: #111111;
        border-radius: 10px;
        border: 1px solid #333333;
    }
    
    /* Chart container styling */
    .stPlotlyChart {
        background: #111111;
        border-radius: 15px;
        border: 1px solid #333333;
        padding: 1rem;
    }
    
    /* Warning and error messages */
    .stAlert {
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* Success messages */
    .stSuccess {
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* Info messages */
    .stInfo {
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555555;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Linear Regression Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Upload Data", "Analysis", "Visualizations", "Predictions", "Model Management"]
    )
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LinearRegressionAnalyzer()
    
    if page == "Home":
        show_home_page()
    elif page == "Upload Data":
        show_upload_page()
    elif page == "Analysis":
        show_analysis_page()
    elif page == "Visualizations":
        show_visualizations_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "Model Management":
        show_model_management_page()

def show_home_page():
    st.markdown("""
    <div class="info-box">
        <h2>Welcome to Linear Regression Analysis Tool</h2>
        <p>A comprehensive web-based tool for performing linear regression analysis with Excel and CSV files.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Data Handling</h3>
            <ul>
                <li>Support for Excel (.xlsx, .xls) and CSV files</li>
                <li>Interactive variable selection</li>
                <li>Automatic categorical feature detection</li>
                <li>One-hot encoding for categorical variables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>Comprehensive Metrics</h3>
            <ul>
                <li>R² Score and Adjusted R² Score</li>
                <li>RMSE, MSE, and MAE</li>
                <li>Model coefficients and feature importance</li>
                <li>Training and test set evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Visualizations</h3>
            <ul>
                <li>Interactive correlation matrix</li>
                <li>Actual vs Predicted plots</li>
                <li>Residual analysis</li>
                <li>Feature importance visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Prediction Capabilities</h3>
            <ul>
                <li>New data prediction</li>
                <li>Batch prediction support</li>
                <li>Model persistence</li>
                <li>Real-time results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>Quick Start</h3>
        <p>1. Go to "Upload Data" to load your dataset</p>
        <p>2. Navigate to "Analysis" to configure and train your model</p>
        <p>3. View "Visualizations" for insights</p>
        <p>4. Use "Predictions" for new data analysis</p>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    st.header("📁 Upload Your Data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your dataset in Excel or CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            
            st.session_state.data = data
            st.session_state.analyzer.data = data
            
            st.success(f"Data loaded successfully! Shape: {data.shape}")
            
            # Display data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
            
            with col2:
                st.subheader("Data Information")
                st.write(f"**Rows:** {data.shape[0]}")
                st.write(f"**Columns:** {data.shape[1]}")
                st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
                
                # Data types
                st.write("**Data Types:**")
                for col, dtype in data.dtypes.items():
                    st.write(f"  - {col}: {dtype}")
            
            # Missing values analysis
            if data.isnull().sum().sum() > 0:
                st.subheader("Missing Values Analysis")
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                st.bar_chart(missing_data)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Sample data option
    st.subheader("Or Use Sample Data")
    if st.button("Generate Sample Dataset"):
        sample_data = create_sample_data()
        st.session_state.data = sample_data
        st.session_state.analyzer.data = sample_data
        st.success("Sample data generated successfully!")
        st.dataframe(sample_data.head(), use_container_width=True)

def show_analysis_page():
    st.header("Model Analysis")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload data first in the 'Upload Data' page.")
        return
    
    data = st.session_state.data
    analyzer = st.session_state.analyzer
    
    # Step 1: Select dependent variable
    st.subheader("1. Select Dependent Variable")
    dependent_var = st.selectbox(
        "Choose your target variable:",
        data.columns,
        help="Select the variable you want to predict"
    )
    
    if dependent_var:
        analyzer.dependent_variable = dependent_var
        st.success(f"Dependent variable set to: {dependent_var}")
    
    # Step 2: Handle Missing Values - ENHANCED VERSION
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <h2 style="color: white; text-align: center; margin: 0; font-size: 24px;">Missing Value Detection & Treatment</h2>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">Advanced AI-powered missing value analysis and imputation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for missing values with enhanced visualization
    missing_counts = data.isnull().sum()
    total_missing = missing_counts.sum()
    total_cells = len(data) * len(data.columns)
    missing_percentage = (total_missing / total_cells) * 100
    
    # Create a beautiful missing value dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(255,107,107,0.3);">
            <h3 style="color: white; margin: 0; font-size: 18px;">Total Missing</h3>
            <p style="color: white; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{total_missing:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(78,205,196,0.3);">
            <h3 style="color: white; margin: 0; font-size: 18px;">Missing %</h3>
            <p style="color: white; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{missing_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #45b7d1, #96c93d); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(69,183,209,0.3);">
            <h3 style="color: white; margin: 0; font-size: 18px;">Total Cells</h3>
            <p style="color: white; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{total_cells:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        affected_columns = len(missing_counts[missing_counts > 0])
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(240,147,251,0.3);">
            <h3 style="color: white; margin: 0; font-size: 18px;">Affected Columns</h3>
            <p style="color: white; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{affected_columns}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if total_missing > 0:
        # Enhanced missing values visualization
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid rgba(255,255,255,0.1);">
            <h3 style="color: #ffffff; margin: 0 0 15px 0;">Detailed Missing Value Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create enhanced missing values summary with progress bars
        missing_data = []
        for col in missing_counts[missing_counts > 0].index:
            count = missing_counts[col]
            percentage = (count / len(data)) * 100
            missing_data.append({
                'Column': col,
                'Missing Count': count,
                'Missing %': percentage,
                'Data Type': str(data[col].dtype),
                'Severity': 'Critical' if percentage > 20 else 'Moderate' if percentage > 5 else 'Low'
            })
        
        missing_summary = pd.DataFrame(missing_data)
        
        # Display with enhanced styling
        st.dataframe(
            missing_summary,
            use_container_width=True,
            column_config={
                "Missing %": st.column_config.ProgressColumn(
                    "Missing %",
                    help="Percentage of missing values",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
                "Severity": st.column_config.TextColumn(
                    "Severity",
                    help="Impact level of missing values"
                )
            }
        )
        
        # AI-powered method recommendation
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <h3 style="color: white; margin: 0 0 15px 0;">AI Method Recommendation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced method selection with descriptions
        method_descriptions = {
            "Auto-detect": "AI analyzes your data and chooses the optimal method automatically",
"Mean": "Fills numerical missing values with column average (good for normal distributions)",
"Median": "Fills numerical missing values with column median (robust to outliers)",
"Mode": "Fills categorical missing values with most frequent value",
"Drop rows": "Removes rows with any missing values (use when missing data is minimal)",
"Interpolate": "Linear interpolation for numerical data (excellent for time series)",
"KNN": "Advanced: Uses K-Nearest Neighbors to predict missing values intelligently"
        }
        
        missing_method = st.selectbox(
            "Choose your missing value treatment strategy:",
            list(method_descriptions.keys()),
            help="Select the method that best fits your data characteristics"
        )
        
        # Show method description
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #667eea;">
            <p style="color: #ffffff; margin: 0; font-style: italic;">{method_descriptions[missing_method]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Map method names to analyzer method names
        method_mapping = {
            "Auto-detect": "auto",
            "Mean": "mean", 
            "Median": "median",
            "Mode": "mode",
            "Drop rows": "drop",
            "Interpolate": "interpolate",
            "KNN": "knn"
        }
        
        analyzer.missing_value_method = method_mapping[missing_method]
        
        # Show what will happen
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); padding: 15px; border-radius: 10px; margin: 15px 0; box-shadow: 0 4px 15px rgba(78,205,196,0.3);">
            <h4 style="color: white; margin: 0 0 10px 0;">Action Plan</h4>
            <p style="color: white; margin: 0; opacity: 0.9;">Will apply <strong>{missing_method}</strong> method to handle {total_missing:,} missing values across {affected_columns} columns</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(78,205,196,0.3); text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 24px;">Perfect Data Quality!</h3>
            <p style="color: white; margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">No missing values detected in your dataset. Your data is ready for analysis!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 3: Identify categorical features
    st.subheader("3. Identify Categorical Features")
    available_features = [col for col in data.columns if col != dependent_var]
    
    categorical_features = st.multiselect(
        "Select categorical features:",
        available_features,
        help="Choose features that represent categories (e.g., Education, Department)"
    )
    
    # Always set both categorical and numerical features
    analyzer.categorical_features = categorical_features
    analyzer.numerical_features = [col for col in available_features if col not in categorical_features]
    
    if categorical_features:
        st.success(f"Categorical features: {categorical_features}")
        
        # Show available categories if model is trained
        if hasattr(analyzer, 'categorical_categories') and analyzer.categorical_categories:
            st.subheader("Available Categories for Prediction")
            for feature, categories in analyzer.categorical_categories.items():
                st.write(f"**{feature}:** {', '.join(categories)}")
    else:
        st.info("No categorical features selected")
    
    st.info(f"Numerical features: {analyzer.numerical_features}")
    
    # Step 4: Advanced Preprocessing Configuration
    st.subheader("4. Advanced Preprocessing Configuration")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <h3 style="color: white; margin: 0; text-align: center;">Advanced Data Preprocessing</h3>
        <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">Configure scaling, outlier detection, feature selection, and transformations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different preprocessing options
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scaling", "Outliers", "Feature Selection", "Transformations", "Feature Engineering"])
    
    with tab1:
        st.markdown("### Scaling Configuration")
        scaling_method = st.selectbox(
            "Choose scaling method:",
            ["standard", "minmax", "robust", "normalizer", "none"],
            format_func=lambda x: {
                "standard": "StandardScaler (mean=0, std=1)",
                "minmax": "MinMaxScaler (0 to 1)",
                "robust": "RobustScaler (robust to outliers)",
                "normalizer": "Normalizer (unit norm)",
                "none": "No scaling"
            }[x],
            help="Select how to scale your numerical features"
        )
        analyzer.preprocessing_config['scaling_method'] = scaling_method
        st.info(f"Selected scaling: {scaling_method}")
    
    with tab2:
        st.markdown("### Outlier Detection & Handling")
        outlier_detection = st.selectbox(
            "Outlier detection method:",
            ["none", "iqr", "zscore", "isolation_forest"],
            format_func=lambda x: {
                "none": "No outlier detection",
                "iqr": "IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)",
                "zscore": "Z-score method (|z| > 3)",
                "isolation_forest": "Isolation Forest"
            }[x]
        )
        analyzer.preprocessing_config['outlier_detection'] = outlier_detection
        
        if outlier_detection != "none":
            outlier_handling = st.selectbox(
                "Outlier handling method:",
                ["remove", "cap", "transform"],
                format_func=lambda x: {
                    "remove": "Remove outliers",
                    "cap": "Cap outliers (winsorize)",
                    "transform": "Transform outliers (log transform)"
                }[x]
            )
            analyzer.preprocessing_config['outlier_handling'] = outlier_handling
            st.info(f"Detection: {outlier_detection}, Handling: {outlier_handling}")
    
    with tab3:
        st.markdown("### Feature Selection")
        feature_selection = st.selectbox(
            "Feature selection method:",
            ["none", "variance", "correlation", "kbest", "rfe"],
            format_func=lambda x: {
                "none": "No feature selection",
                "variance": "Variance threshold (remove low variance)",
                "correlation": "Correlation-based (remove highly correlated)",
                "kbest": "K-best features (F-statistic)",
                "rfe": "Recursive Feature Elimination (RFE)"
            }[x]
        )
        analyzer.preprocessing_config['feature_selection'] = feature_selection
        
        if feature_selection == "variance":
            threshold = st.slider("Variance threshold:", 0.0, 0.1, 0.01, 0.001)
            analyzer.preprocessing_config['feature_selection_params'] = {'threshold': threshold}
        elif feature_selection == "correlation":
            threshold = st.slider("Correlation threshold:", 0.7, 1.0, 0.95, 0.01)
            analyzer.preprocessing_config['feature_selection_params'] = {'threshold': threshold}
        elif feature_selection in ["kbest", "rfe"]:
            k = st.slider("Number of features to select:", 1, min(20, len(analyzer.numerical_features)), 
                         min(10, len(analyzer.numerical_features)))
            param_name = 'k' if feature_selection == "kbest" else 'n_features'
            analyzer.preprocessing_config['feature_selection_params'] = {param_name: k}
        
        if feature_selection != "none":
            st.info(f"Feature selection: {feature_selection}")
    
    with tab4:
        st.markdown("### Data Transformations")
        data_transformation = st.selectbox(
            "Data transformation method:",
            ["none", "log", "boxcox"],
            format_func=lambda x: {
                "none": "No transformation",
                "log": "Log transformation (log1p)",
                "boxcox": "Box-Cox transformation"
            }[x]
        )
        analyzer.preprocessing_config['data_transformation'] = data_transformation
        if data_transformation != "none":
            st.info(f"Transformation: {data_transformation}")
    
    with tab5:
        st.markdown("### Feature Engineering")
        col1, col2 = st.columns(2)
        
        with col1:
            polynomial_features = st.checkbox("Add polynomial features", value=False)
            analyzer.preprocessing_config['polynomial_features'] = polynomial_features
            
            if polynomial_features:
                degree = st.slider("Polynomial degree:", 2, 3, 2)
                if 'feature_selection_params' not in analyzer.preprocessing_config:
                    analyzer.preprocessing_config['feature_selection_params'] = {}
                analyzer.preprocessing_config['feature_selection_params']['poly_degree'] = degree
        
        with col2:
            interaction_features = st.checkbox("Add interaction features", value=False)
            analyzer.preprocessing_config['interaction_features'] = interaction_features
        
        if polynomial_features or interaction_features:
            st.info("Feature engineering enabled")
    
    # Step 5: Set Dependent Variable Range
    st.subheader("5. Set Dependent Variable Range")
    
    # Show current data statistics
    current_min = data[dependent_var].min()
    current_max = data[dependent_var].max()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Minimum", f"{current_min:.2f}")
    with col2:
        st.metric("Current Maximum", f"{current_max:.2f}")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: #ffffff; margin: 0 0 10px 0;">Range Validation</h4>
        <p style="color: #cccccc; margin: 0; font-size: 14px;">
            Set a maximum realistic value for your dependent variable. This will prevent the model from making 
            unrealistic predictions when given extreme input values. Predictions exceeding this limit will be 
            automatically capped to the maximum value.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    max_value = st.number_input(
        f"Maximum realistic value for {dependent_var}:",
        min_value=float(current_min),
        max_value=float(current_max * 10),  # Allow up to 10x current max
        value=float(current_max),
        step=0.01,
        help="Enter the maximum realistic value your dependent variable can reach"
    )
    
    # Store the range in the analyzer
    analyzer.dependent_variable_range = max_value
    
    if max_value > current_max:
        st.warning(f"⚠️ The maximum value ({max_value:.2f}) is higher than the current data maximum ({current_max:.2f}). This allows for future growth predictions.")
    elif max_value < current_max:
        st.warning(f"⚠️ The maximum value ({max_value:.2f}) is lower than the current data maximum ({current_max:.2f}). This will cap predictions to a more conservative range.")
    else:
        st.success(f"✅ Maximum range set to current data maximum: {max_value:.2f}")
    
    # Step 6: Train model
    st.subheader("6. Train Model")
    if st.button("Train Linear Regression Model", type="primary"):
        if not hasattr(analyzer, 'dependent_variable') or analyzer.dependent_variable is None:
            st.error("Please select a dependent variable first.")
            return
        
        with st.spinner("Training model..."):
            try:
                # Preprocess data with selected missing value method
                if hasattr(analyzer, 'missing_value_method'):
                    success = analyzer.preprocess_data(missing_method=analyzer.missing_value_method)
                else:
                    success = analyzer.preprocess_data()
                
                if success:
                    st.success("Data preprocessing completed!")
                    
                    # Train model
                    if analyzer.train_model():
                        st.success("Model training completed!")
                        
                        # Calculate metrics
                        metrics = analyzer.calculate_metrics()
                        if metrics:
                            st.session_state.metrics = metrics
                            st.success("Metrics calculated successfully!")
                            
                            # Display metrics
                            display_metrics(metrics)
                            
                            # Display categorical categories for prediction reference
                            if hasattr(analyzer, 'categorical_categories') and analyzer.categorical_categories:
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
                                    <h3 style="color: white; margin: 0; text-align: center;">Categorical Categories for Predictions</h3>
                                    <p style="color: white; text-align: center; margin: 10px 0 0 0; opacity: 0.9;">Use these exact values when making predictions</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for feature, categories in analyzer.categorical_categories.items():
                                    st.markdown(f"""
                                    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1);">
                                        <h4 style="color: #ffffff; margin: 0 0 10px 0;">{feature}</h4>
                                        <p style="color: #cccccc; margin: 0; font-family: monospace;">{', '.join(categories)}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to calculate metrics.")
                    else:
                        st.error("Model training failed.")
                else:
                    st.error("Data preprocessing failed.")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def display_metrics(metrics):
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Set Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <h4>R² Score</h4>
            <h2>{metrics['train_r2']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Adjusted R² Score</h4>
            <h2>{metrics['train_adj_r2']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>RMSE</h4>
            <h2>{metrics['train_rmse']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Test Set Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <h4>R² Score</h4>
            <h2>{metrics['test_r2']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Adjusted R² Score</h4>
            <h2>{metrics['test_adj_r2']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>RMSE</h4>
            <h2>{metrics['test_rmse']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

def show_visualizations_page():
    st.header("Visualizations")
    
    if 'metrics' not in st.session_state:
        st.warning("Please train a model first in the 'Analysis' page.")
        return
    
    analyzer = st.session_state.analyzer
    
    # Visualization options
    viz_option = st.selectbox(
        "Choose visualization:",
        ["Correlation Matrix", "Actual vs Predicted", "Residuals", "Feature Importance"]
    )
    
    if viz_option == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        if hasattr(analyzer, 'data') and analyzer.data is not None:
            numerical_data = analyzer.data.select_dtypes(include=[np.number])
            correlation_matrix = numerical_data.corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Correlation Matrix (Numerical Variables)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Actual vs Predicted":
        st.subheader("Actual vs Predicted")
        if hasattr(analyzer, 'y_train') and hasattr(analyzer, 'y_train_pred'):
            col1, col2 = st.columns(2)
            
            with col1:
                fig_train = px.scatter(
                    x=analyzer.y_train,
                    y=analyzer.y_train_pred,
                    title="Training: Actual vs Predicted",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                fig_train.add_trace(go.Scatter(
                    x=[analyzer.y_train.min(), analyzer.y_train.max()],
                    y=[analyzer.y_train.min(), analyzer.y_train.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig_train, use_container_width=True)
            
            with col2:
                fig_test = px.scatter(
                    x=analyzer.y_test,
                    y=analyzer.y_test_pred,
                    title="Test: Actual vs Predicted",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                fig_test.add_trace(go.Scatter(
                    x=[analyzer.y_test.min(), analyzer.y_test.max()],
                    y=[analyzer.y_test.min(), analyzer.y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig_test, use_container_width=True)
    
    elif viz_option == "Residuals":
        st.subheader("Residuals Analysis")
        if hasattr(analyzer, 'y_train_pred'):
            col1, col2 = st.columns(2)
            
            with col1:
                residuals_train = analyzer.y_train - analyzer.y_train_pred
                fig_res_train = px.scatter(
                    x=analyzer.y_train_pred,
                    y=residuals_train,
                    title="Training: Residuals vs Predicted",
                    labels={'x': 'Predicted Values', 'y': 'Residuals'}
                )
                fig_res_train.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res_train, use_container_width=True)
            
            with col2:
                residuals_test = analyzer.y_test - analyzer.y_test_pred
                fig_res_test = px.scatter(
                    x=analyzer.y_test_pred,
                    y=residuals_test,
                    title="Test: Residuals vs Predicted",
                    labels={'x': 'Predicted Values', 'y': 'Residuals'}
                )
                fig_res_test.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res_test, use_container_width=True)
    
    elif viz_option == "Feature Importance":
        st.subheader("Feature Importance")
        if hasattr(analyzer, 'model') and analyzer.model is not None:
            coefficients = pd.DataFrame({
                'Feature': analyzer.feature_names,
                'Coefficient': analyzer.model.coef_
            })
            coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(
                coefficients,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title="Feature Importance (Linear Regression Coefficients)",
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_predictions_page():
    st.header("Make Predictions")
    
    if not hasattr(st.session_state.analyzer, 'model') or st.session_state.analyzer.model is None:
        st.warning("Please train a model first in the 'Analysis' page.")
        return
    
    analyzer = st.session_state.analyzer
    
    # Prediction options
    pred_option = st.selectbox(
        "Choose prediction method:",
        ["Upload New Data File", "Manual Data Entry"]
    )
    
    if pred_option == "Upload New Data File":
        st.subheader("Upload New Data for Prediction")
        new_file = st.file_uploader(
            "Choose a file with new data:",
            type=['xlsx', 'xls', 'csv']
        )
        
        if new_file:
            try:
                if new_file.name.endswith(('.xlsx', '.xls')):
                    new_data = pd.read_excel(new_file)
                else:
                    new_data = pd.read_csv(new_file)
                
                # Remove dependent variable if present
                if analyzer.dependent_variable in new_data.columns:
                    new_data = new_data.drop(columns=[analyzer.dependent_variable])
                
                st.success(f"New data loaded! Shape: {new_data.shape}")
                st.dataframe(new_data.head(), use_container_width=True)
                
                if st.button("Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        result = analyzer.predict_new_data(new_data)
                        if result:
                            new_data_with_predictions, predictions = result
                            st.success("Predictions completed!")
                            
                            # Check if any predictions were capped
                            if hasattr(analyzer, 'dependent_variable_range') and analyzer.dependent_variable_range:
                                capped_count = sum(1 for pred in predictions if pred > analyzer.dependent_variable_range)
                                if capped_count > 0:
                                    st.warning(f"⚠️ {capped_count} prediction(s) were capped to the maximum range of {analyzer.dependent_variable_range:.2f}")
                            
                            st.subheader("Prediction Results")
                            st.dataframe(new_data_with_predictions, use_container_width=True)
                            
                            # Download results
                            csv = new_data_with_predictions.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("Failed to make predictions.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif pred_option == "Manual Data Entry":
        st.subheader("Manual Data Entry")
        
        if hasattr(analyzer, 'data') and analyzer.data is not None:
            features = [col for col in analyzer.data.columns if col != analyzer.dependent_variable]
            
            # Create input fields
            input_data = {}
            cols = st.columns(2)
            
            for i, feature in enumerate(features):
                with cols[i % 2]:
                     if feature in analyzer.numerical_features:
                         input_data[feature] = st.number_input(
                             f"{feature}:",
                             value=0.0,
                             step=0.1
                         )
                     else:
                         # Show available categories for categorical features
                         if hasattr(analyzer, 'categorical_categories') and feature in analyzer.categorical_categories:
                             categories = analyzer.categorical_categories[feature]
                             st.markdown(f"**{feature}** (Categorical)")
                             st.info(f"Available categories: {', '.join(categories)}")
                             input_data[feature] = st.selectbox(
                                 f"Select {feature}:",
                                 options=categories,
                                 help=f"Choose from the available categories in your dataset"
                             )
                         else:
                             st.warning(f"⚠️ {feature} is categorical but categories not found. Please retrain the model.")
                             input_data[feature] = st.text_input(f"{feature} (enter exact value from dataset):")
            
            if st.button("Predict", type="primary"):
                if all(input_data.values()):
                    try:
                        new_data = pd.DataFrame([input_data])
                        result = analyzer.predict_new_data(new_data)
                        if result:
                            new_data_with_predictions, predictions = result
                            st.success("Prediction completed!")
                            
                            # Check if prediction was capped
                            original_prediction = predictions[0]
                            if hasattr(analyzer, 'dependent_variable_range') and analyzer.dependent_variable_range:
                                if original_prediction > analyzer.dependent_variable_range:
                                    st.warning(f"⚠️ Prediction was capped from {original_prediction:.2f} to {analyzer.dependent_variable_range:.2f} due to range validation.")
                                    st.markdown(f"""
                                    <div class="success-box">
                                        <h3>Prediction Result (Capped)</h3>
                                        <h2>{analyzer.dependent_variable}: {analyzer.dependent_variable_range:.2f}</h2>
                                        <p style="font-size: 14px; opacity: 0.8;">Original prediction: {original_prediction:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="success-box">
                                        <h3>Prediction Result</h3>
                                        <h2>{analyzer.dependent_variable}: {predictions[0]:.2f}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="success-box">
                                    <h3>Prediction Result</h3>
                                    <h2>{analyzer.dependent_variable}: {predictions[0]:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to make prediction.")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                else:
                    st.warning("Please fill in all fields.")

def show_model_management_page():
    st.header("Model Management")
    
    analyzer = st.session_state.analyzer
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save Model")
        if hasattr(analyzer, 'model') and analyzer.model is not None:
            model_name = st.text_input("Model name:", value="my_model.pkl")
            if st.button("Save Model", type="primary"):
                try:
                    analyzer.save_model(model_name)
                    st.success(f"Model saved as {model_name}")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        else:
            st.warning("No trained model available.")
    
    with col2:
        st.subheader("Load Model")
        uploaded_model = st.file_uploader(
            "Choose a saved model file:",
            type=['pkl']
        )
        
        if uploaded_model:
            if st.button("Load Model", type="primary"):
                try:
                    # Save uploaded file temporarily
                    with open("temp_model.pkl", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    analyzer.load_model("temp_model.pkl")
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

def create_sample_data():
    """Create sample data for demonstration"""
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
    
    # Add some missing values for testing
    # Add missing values to numerical columns
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data.loc[missing_indices, 'Age'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data.loc[missing_indices, 'Experience'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    data.loc[missing_indices, 'Performance_Score'] = np.nan
    
    # Add missing values to categorical columns
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    data.loc[missing_indices, 'Education'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data.loc[missing_indices, 'Department'] = np.nan
    
    return data

if __name__ == "__main__":
    main() 