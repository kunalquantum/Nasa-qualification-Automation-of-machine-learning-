import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config(page_title='Autopredictor', page_icon=':infinity:', initial_sidebar_state='collapsed')
st.title(' :earth_africa: AUTO PREDICTOR ')
st.caption("We visualize, predict, and test")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    st.spinner()
    data = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.header("Dataset Preview")
    st.write(data.head())

    # Choose the target column
    target_column = st.multiselect("Select the target column", data.columns)
    
    # Choose the dependent variable
    dependent_column = st.multiselect("Select the dependent column", data.columns)

    if target_column and dependent_column:
        # Data Preprocessing
        st.header("Data Preprocessing")
        st.write("Performing data preprocessing...")
        st.progress(0)

        # Separate categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object']).columns

        # Apply label encoding to categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

        # Fill missing values with dummy values (you can customize this)
        data.fillna(0, inplace=True)

        # Drop unwanted columns
        unwanted_columns = st.multiselect("Select columns to drop", data.columns)
        data.drop(unwanted_columns, axis=1, inplace=True)

        # Save the preprocessed dataset
        preprocessed_data_file = "preprocessed_data.csv"
        data.to_csv(preprocessed_data_file, index=False)
        st.write(f"Preprocessed data saved to {preprocessed_data_file}")

        # Add download button for preprocessed data
        st.markdown(f"Download the Preprocessed Data: [preprocessed_data.csv](./{preprocessed_data_file})")

        # Split data into X and y
        X = data.drop(target_column, axis=1)
        y = data[target_column[0]]  # Assuming a single target column

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        st.header("Model Training")
        with st.spinner("Training the model..."):
            try:
                # Attempt to train a classifier model
                classifier_model = RandomForestClassifier()
                classifier_model.fit(X_train, y_train)
                st.success("Classification model training complete!")

                # Save the trained model
                classifier_model_file = "classifier_model.pkl"
                joblib.dump(classifier_model, classifier_model_file)
                st.write(f"Classification model saved to {classifier_model_file}")

                # Add download button for the trained classification model
                st.markdown(f"Download the Trained Classification Model: [classifier_model.pkl](./{classifier_model_file})")

                # Model evaluation
                y_pred_classifier = classifier_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_classifier)
                st.write(f"Classification Model Accuracy: {accuracy}")

                # Visualization
                st.write("Feature Importance (Classification Model):")

                # Visualize histograms for target and dependent variable
                st.header("Histograms")
                st.write("Histogram of Target Variable:")
                plt.figure(figsize=(8, 6))
                sns.histplot(y, kde=True)
                st.pyplot()

                st.write(f"Histogram of Dependent Variable ({dependent_column[0]}):")
                plt.figure(figsize=(8, 6))
                sns.histplot(data[dependent_column[0]], kde=True)
                st.pyplot()

            except ValueError:
                # If a classifier model cannot be trained, attempt to train a regression model
                st.warning("Cannot train a classification model. Attempting to train a regression model...")
                try:
                    # Train a regression model
                    regression_model = RandomForestRegressor()
                    regression_model.fit(X_train, y_train)
                    st.success("Regression model training complete!")

                    # Save the trained model
                    regression_model_file = "regression_model.pkl"
                    joblib.dump(regression_model, regression_model_file)
                    st.write(f"Regression model saved to {regression_model_file}")

                    # Add download button for the trained regression model
                    st.markdown(f"Download the Trained Regression Model: [regression_model.pkl](./{regression_model_file})")

                    # Model evaluation
                    y_pred_regression = regression_model.predict(X_test)
                    r2 = r2_score(y_test, y_pred_regression)
                    st.write(f"Regression Model R2 Score: {r2}")

                    # Visualization
                    st.write("Feature Importance (Regression Model):")

                    # Visualize histograms for target and dependent variable
                    st.header("Histograms")
                    st.write("Histogram of Target Variable:")
                    plt.figure(figsize=(8, 6))
                    sns.histplot(y, kde=True)
                    st.pyplot()

                    st.write(f"Histogram of Dependent Variable ({dependent_column[0]}):")
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data[dependent_column[0]], kde=True)
                    st.pyplot()

                except ValueError:
                    st.error("Cannot train a regression model. Please select different columns.")
