import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, VotingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib
import shap

# Title of the app
st.title("Advanced Ensemble Learning with Customization and Visualization")

# Upload CSV data
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Check if data is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sidebar options
    st.sidebar.subheader("Select Features and Target")
    target = st.sidebar.selectbox("Select the Target Column", df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns", df.columns)

    # Check if at least one feature is selected
    if len(feature_cols) > 0:
        # Ensure the target variable is numeric (assuming it's a regression task)
        if not np.issubdtype(df[target].dtype, np.number):
            st.warning("Target variable is not numeric. Please check your data.")
        else:
            X = df[feature_cols]
            y = df[target]

            # Data preprocessing options
            preprocessing_options = st.sidebar.checkbox("Data Preprocessing Options")
            if preprocessing_options:
                # Handle missing values
                imputer_strategy = st.sidebar.selectbox("Imputation Strategy", ["None", "Mean", "Median", "Most Frequent"])
                if imputer_strategy != "None":
                    imputer = SimpleImputer(strategy=imputer_strategy)
                else:
                    imputer = None

                # Encode categorical features
                cat_features = X.select_dtypes(include=['object']).columns
                if len(cat_features) > 0:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                else:
                    encoder = None

                # Scale numerical features
                num_features = X.select_dtypes(include=['number']).columns
                if len(num_features) > 0:
                    scaler = StandardScaler()
                else:
                    scaler = None

                transformers = []
                if imputer:
                    transformers.append(("imputer", imputer, X.select_dtypes(include=['number'])))
                if encoder:
                    transformers.append(("encoder", encoder, cat_features))
                if scaler:
                    transformers.append(("scaler", scaler, num_features))

                preprocessor = ColumnTransformer(transformers)

                X = preprocessor.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Check if the target variable is for classification or regression
            classification_task = len(np.unique(y)) <= 2
            if classification_task:
                # For classification tasks
                ensemble_models = {
                    "Random Forest": RandomForestClassifier(),
                    "AdaBoost": AdaBoostClassifier(),
                    "Voting Classifier": VotingClassifier(estimators=[
                        ('rf', RandomForestClassifier()),
                        ('ada', AdaBoostClassifier())
                    ]),
                    "SVM": SVC(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier()
                }
            else:
                # For regression tasks
                ensemble_models = {
                    "Random Forest": RandomForestRegressor(),
                    "AdaBoost": AdaBoostRegressor(),
                    "Voting Regressor": VotingClassifier(estimators=[
                        ('rf', RandomForestRegressor()),
                        ('ada', AdaBoostRegressor())
                    ]),
                    "SVR": SVR(),
                    "K-Nearest Neighbors": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor()
                }

            # Lists to store model names and their accuracies
            model_names = []
            accuracies = []

            # Training, evaluating, and visualizing all models
            for model_name, model in ensemble_models.items():
                try:
                    # Train the selected ensemble model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Calculate accuracy for classification or MSE for regression
                    if classification_task:
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy of {model_name}: {accuracy:.2f}")

                        # Visualize accuracy
                        st.subheader(f"Accuracy Visualization - {model_name}")
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(len(y_test)), y_test, label="Actual", linestyle="-", color="blue")
                        plt.plot(range(len(y_pred)), y_pred, label="Predicted", linestyle="--", color="orange")
                        plt.xlabel("Samples")
                        plt.ylabel("Values")
                        plt.title(f"Accuracy Visualization - {model_name}")
                        plt.legend()
                        st.pyplot()

                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        st.write(f"Mean Squared Error of {model_name}: {mse:.2f}")

                        # Visualize loss (MSE)
                        st.subheader(f"Loss (MSE) Visualization - {model_name}")
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(len(y_test)), y_test, label="Actual", linestyle="-", color="blue")
                        plt.xlabel("Samples")
                        plt.ylabel("Values")
                        plt.title(f"Loss (MSE) Visualization - {model_name}")
                        plt.legend()
                        st.pyplot()

                        # Store model name and accuracy for the table
                        model_names.append(model_name)
                        accuracies.append(accuracy if classification_task else mse)

                except Exception as e:
                    st.error(f"An error occurred for {model_name}: {str(e)}")

            # Create a table to display model names and accuracies
            st.subheader("Model Accuracies")
            df_accuracies = pd.DataFrame({"Model": model_names, "Accuracy (MSE for Regression)": accuracies})
            st.table(df_accuracies)

           

    else:
        st.warning("Please select at least one feature column.")
else:
    st.warning("Please upload a CSV file.")
