import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get the current date and time as a string
def get_current_datetime():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Streamlit UI
st.title("Association Rule Learning App")
st.sidebar.header("Settings")

# Display the last updated timestamp
st.sidebar.markdown(f"Last Updated: {get_current_datetime()}")

# Upload CSV data
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.markdown("### Data Preview")
    data_preview = pd.read_csv(uploaded_file)
    st.sidebar.write(data_preview.head())

    min_support = st.sidebar.slider("Minimum Support", 0.0, 1.0, 0.1, help="Set the minimum support threshold.")
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, help="Set the minimum confidence threshold.")

    # Perform Association Rule Mining
    st.header("Association Rules")
    st.subheader("Frequent Item Sets")

    # Preprocess the data (convert to binary format)
    def binarize(x):
        if x <= 0:
            return 0
        else:
            return 1

    data = data_preview.applymap(binarize)

    # Apply Apriori algorithm
    st.text("Performing Apriori algorithm...")
    frequent_item_sets = apriori(data, min_support=min_support, use_colnames=True)

    st.write("Frequent item sets found:")
    st.write(frequent_item_sets)

    st.subheader("Association Rules")

    # Generate association rules
    st.text("Generating association rules...")
    rules = association_rules(frequent_item_sets, metric="lift", min_threshold=min_confidence)

    st.write("All association rules:")
    st.write(rules)

    # Filter association rules based on user input
    st.sidebar.markdown("### Filter Rules")
    min_lift = st.sidebar.slider("Minimum Lift", 0.0, 10.0, 1.0, help="Set the minimum lift threshold.")
    filtered_rules = rules[rules["lift"] >= min_lift]

    st.header("Filtered Association Rules")
    st.write("Filtered association rules:")
    st.write(filtered_rules)

    # Visualizations

    # Histogram for Support Threshold
    st.sidebar.markdown("### Support Threshold Distribution")
    plt.hist(frequent_item_sets['support'], bins=20, edgecolor='k')
    st.sidebar.pyplot()

    # Scatter Plot for Support vs. Confidence
    st.subheader("Association Rules - Support vs. Confidence")
    sns.scatterplot(data=filtered_rules, x='support', y='confidence')
    st.pyplot()

    # Heatmap for Lift Values
    st.subheader("Association Rules - Lift Heatmap")
    lift_matrix = rules.pivot(index='antecedents', columns='consequents', values='lift')
    sns.heatmap(lift_matrix, cmap="viridis", linewidths=0.5, linecolor='white')
    st.pyplot()

# Provide explanations and hints
st.sidebar.markdown("### How to Use")
st.sidebar.write("1. Upload a CSV file containing transaction data.")
st.sidebar.write("2. Adjust the minimum support and confidence thresholds.")
st.sidebar.write("3. Filter rules based on lift using the slider.")