import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="Multivariate Model Training and Evaluation",
    page_icon="✅",
    layout="wide",
)

# Streamlit app title and description
st.title("Multivariate Model Training and Evaluation")
st.subheader("Upload your dataset, choose a target variable, and compare regression models.")

# Data upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.sidebar.subheader("Preview of the uploaded dataset:")
    st.sidebar.write(data.head())
    
    # Target variable selection
    st.sidebar.subheader("Select the target variable:")
    target_variable = st.sidebar.selectbox("Choose the target variable", data.columns)
    
    if st.sidebar.button("Start Data Cleaning and Model Evaluation"):
        # Data Cleaning
        st.header("Data Cleaning:")
        
        # Handle missing values
        st.text("Step 1: Handling missing values...")
        time.sleep(2)
        imputer = SimpleImputer(strategy='mean')
        data[target_variable] = imputer.fit_transform(data[target_variable].values.reshape(-1, 1))
        st.write("Missing values filled with mean.")
        
        # Convert string columns to dummy numerical values (1 to 6 in this example)
        st.text("Step 2: Converting categorical values to numerical...")
        time.sleep(2)
        label_encoder = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = label_encoder.fit_transform(data[col]) % 6 + 1
        st.write("Categorical values converted to numerical (1 to 6).")
        
        # Feature Scaling
        st.text("Step 3: Scaling features...")
        time.sleep(2)
        scaler = StandardScaler()
        data[data.columns.difference([target_variable])] = scaler.fit_transform(data[data.columns.difference([target_variable])])
        st.write("Features scaled using StandardScaler.")
        
        # Split data into features and target variable
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]
        
        # Regression Models
        st.header("Regression Models:")
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "Random Forest Regression": RandomForestRegressor(n_estimators=100)
        }
        
        results = {}
        
        for model_name, model in models.items():
            st.text(f"Training {model_name}...")
            time.sleep(2)
            mse_scores = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
            r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
            
            results[model_name] = {
                "MSE": np.mean(mse_scores),
                "R-squared": np.mean(r2_scores)
            }
        
        # Display results
        st.header("Model Comparison:")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.highlight_min(axis=0))
        
        best_model = results_df["MSE"].idxmin()
        st.write(f"The best model is {best_model} with an MSE of {results_df['MSE'].min():.4f}")
