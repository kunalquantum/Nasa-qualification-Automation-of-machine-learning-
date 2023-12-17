import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title and description of your app
st.title("Advanced Data Cleaning App")
st.write("Upload a dataset, explore the data, apply advanced data cleaning operations, and download the cleaned data.")

# Section 1: Upload Data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Section 2: Data Exploration
st.sidebar.header("Data Exploration")
explore_data = st.sidebar.checkbox("Explore Data", key="explore_data")

# Section 3: Data Cleaning Options
st.sidebar.header("Data Cleaning Options")
remove_null_values = st.sidebar.checkbox("Remove Null Values", key="remove_null_values")
remove_duplicates = st.sidebar.checkbox("Remove Duplicates", key="remove_duplicates")
convert_date_columns = st.sidebar.checkbox("Convert Date Columns to DateTime", key="convert_date_columns")
custom_cleaning = st.sidebar.checkbox("Custom Data Cleaning", key="custom_cleaning")

# Section 4: Data Imputation Strategy (if selected)
imputation_strategy = None
if remove_null_values:
    imputation_strategy = st.sidebar.selectbox("Select Imputation Strategy", ["None", "Mean", "Median", "Mode"], key="imputation_strategy")

# Section 5: Scaling and Normalization
scaling_option = st.sidebar.checkbox("Scale Numeric Columns", key="scaling_option")
normalization_option = st.sidebar.checkbox("Normalize Numeric Columns", key="normalization_option")

# Section 6: Outlier Handling
outlier_option = st.sidebar.checkbox("Handle Outliers", key="outlier_option")

# Section 7: Advanced Outlier Handling (if selected)
outlier_method = None
if outlier_option:
    outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ["None", "Z-score", "Clustering"], key="outlier_method")

# Section 8: Data Visualization
st.sidebar.header("Data Visualization")
visualization_option = st.sidebar.selectbox("Select Visualization", ["None", "Correlation Heatmap", "Scatter Plot"], key="visualization_option")

# Section 9: Data Export Options
st.sidebar.header("Data Export Options")
export_format = st.sidebar.selectbox("Select export format", ["CSV", "Excel"], key="export_format")
export_selected_columns = st.sidebar.checkbox("Export Selected Columns", key="export_selected_columns")

# Section 10: Custom Data Cleaning (if selected)
if custom_cleaning:
    custom_cleaning_code = st.sidebar.text_area("Enter custom cleaning code (Python)", key="custom_cleaning_code")
    st.sidebar.write("Example: df = df.drop(columns=['Column_to_drop'])", key="custom_cleaning_example")

# Section 11: Data Cleaning Process
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the original data
    st.subheader("Original Data")
    if explore_data:
        st.write(df)

    # Data Exploration (Summary Statistics)
    if explore_data:
        st.subheader("Summary Statistics")
        st.write(df.describe())

    # Data Cleaning
    st.subheader("Cleaned Data")

    # Option 1: Remove Null Values
    if remove_null_values:
        if imputation_strategy == "Mean":
            df.fillna(df.mean(), inplace=True)
            st.write("Null values filled with mean.")
        elif imputation_strategy == "Median":
            df.fillna(df.median(), inplace=True)
            st.write("Null values filled with median.")
        elif imputation_strategy == "Mode":
            df.fillna(df.mode().iloc[0], inplace=True)
            st.write("Null values filled with mode.")
        else:
            df.dropna(inplace=True)
            st.write("Null values removed.")
        
        # Show scatter plot after removing null values
        st.subheader("Scatter Plot After Removing Null Values")
        scatter_x = st.selectbox("Select x-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        scatter_y = st.selectbox("Select y-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        plt.figure(figsize=(8, 6))
        plt.scatter(df[scatter_x], df[scatter_y])
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        st.pyplot()

    # Option 2: Remove Duplicates
    if remove_duplicates:
        df.drop_duplicates(inplace=True)
        st.write("Duplicate rows removed.")
        
        # Show scatter plot after removing duplicates
        st.subheader("Scatter Plot After Removing Duplicates")
        scatter_x = st.selectbox("Select x-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        scatter_y = st.selectbox("Select y-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        plt.figure(figsize=(8, 6))
        plt.scatter(df[scatter_x], df[scatter_y])
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        st.pyplot()

    # Option 3: Convert Date Columns to DateTime
    if convert_date_columns:
        date_columns = df.select_dtypes(include=['object']).columns
        for column in date_columns:
            try:
                df[column] = pd.to_datetime(df[column], errors='coerce')
            except:
                st.warning(f"Unable to convert column '{column}' to DateTime.")
        st.write("Date columns converted to DateTime.")
        
    # Option 4: Custom Data Cleaning
    if custom_cleaning and custom_cleaning_code:
        try:
            exec(custom_cleaning_code)
            st.write("Custom data cleaning applied.")
        except Exception as e:
            st.error(f"Error in custom cleaning code: {str(e)}")

    # Option 5: Scaling and Normalization
    if scaling_option:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        st.write("Numeric columns scaled.")
        
    if normalization_option:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
        st.write("Numeric columns normalized.")

    # Option 6: Advanced Outlier Handling
    if outlier_option and outlier_method:
        if outlier_method == "Z-score":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
            df = df[(z_scores < 3).all(axis=1)]  # Keep rows where all z-scores are less than 3
            st.write("Outliers removed using Z-score method.")
            
            # Show scatter plot after removing outliers
            st.subheader("Scatter Plot After Removing Outliers")
            scatter_x = st.selectbox("Select x-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
            scatter_y = st.selectbox("Select y-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
            plt.figure(figsize=(8, 6))
            plt.scatter(df[scatter_x], df[scatter_y])
            plt.xlabel(scatter_x)
            plt.ylabel(scatter_y)
            st.pyplot()

    # Option 7: Data Visualization
    if visualization_option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot()

    elif visualization_option == "Scatter Plot":
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("Select x-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        scatter_y = st.selectbox("Select y-axis (numeric)", df.select_dtypes(include=[np.number]).columns)
        plt.figure(figsize=(8, 6))
        plt.scatter(df[scatter_x], df[scatter_y])
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        st.pyplot()

    # Section 8: Data Export Options
    st.subheader("Data Export Options")
    export_selected_columns = st.checkbox("Export Selected Columns", key="export_selected_columns1")  # Provide a unique key
    export_selected_rows = st.checkbox("Export Selected Rows", key="export_selected_rows1")  # Provide a unique key
    selected_columns = []
    selected_rows = []

    if export_selected_columns:
        selected_columns = st.multiselect("Select columns to export", df.columns)

    if export_selected_rows:
        selected_rows = st.multiselect("Select rows to export", df.index)

    if st.button("Download Cleaned Data"):
        cleaned_data_bytes = io.BytesIO()
        if export_format == "CSV":
            if export_selected_columns:
                df = df[selected_columns]
            if export_selected_rows:
                df = df.loc[selected_rows]
            df.to_csv(cleaned_data_bytes, index=False)
            ext = "csv"
        else:
            if export_selected_columns:
                df = df[selected_columns]
            if export_selected_rows:
                df = df.loc[selected_rows]
            df.to_excel(cleaned_data_bytes, index=False)
            ext = "xlsx"
        cleaned_data_bytes.seek(0)
        st.download_button("Download Cleaned Data", cleaned_data_bytes, f"cleaned_data.{ext}", key="download_button")
