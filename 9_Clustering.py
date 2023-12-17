import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# List of explanatory texts
explanations = [
    "Unsupervised Learning ",
    "Upload a dataset and choose an unsupervised learning algorithm. We'll guide you through the process step by step.",
    "Step 1: Uploading the csv file...",
    "Step 2: Selecting the Algorithm...",
    "Step 3: Customizing the parameters..",
    "Step 4: Training the Model...",
    "Step 5: Visualization complete",
    "Step 6: Conclusion displayed.."
]
clusters = 2
random_state = 123

# Set the title and description
st.title(explanations[0])
st.write(explanations[1])

# Upload dataset
uploaded_file = st.file_uploader(explanations[2], type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Data Cleaning and Handling Missing Values
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    data = data.dropna(subset=categorical_cols)

    data = pd.get_dummies(data, columns=categorical_cols)

    # Sidebar options
    st.sidebar.title(explanations[3])
    algorithm = st.sidebar.selectbox(explanations[4], ["K-Means Clustering", "DBSCAN Clustering", "PCA", "FastICA", "KernelPCA", "t-SNE"])

    if algorithm in ["K-Means Clustering", "DBSCAN Clustering"]:
        st.sidebar.markdown(explanations[5])
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        random_state = st.sidebar.slider("Random State", 0, 100, 42)
    elif algorithm in ["PCA", "FastICA", "KernelPCA"]:
        st.sidebar.markdown(explanations[5])
        n_components = st.sidebar.slider("Number of Components", 1, len(data.columns), 2)
        random_state = st.sidebar.slider("Random State", 0, 100, 42)
    elif algorithm == "t-SNE":
        st.sidebar.markdown(explanations[5])
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
        learning_rate = st.sidebar.slider("Learning Rate", 10, 200, 100)

    # Simulate a generative effect with time.sleep
    for i in range(2, 7):
        st.sidebar.markdown(explanations[i])
        time.sleep(2)  # Add a delay of 2 seconds

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply selected algorithm
    if algorithm == "K-Means Clustering":
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = model.fit_predict(scaled_data)
    elif algorithm == "DBSCAN Clustering":
        eps = 0.5  # You should define eps and min_samples here
        min_samples = 5  # You should define eps and min_samples here
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(scaled_data)
    elif algorithm in ["PCA", "FastICA", "KernelPCA"]:
        if algorithm == "PCA":
            model = PCA(n_components=n_components, random_state=random_state)
        elif algorithm == "FastICA":
            model = FastICA(n_components=n_components, random_state=random_state)
        elif algorithm == "KernelPCA":
            model = KernelPCA(n_components=n_components, kernel='rbf', random_state=random_state)
        reduced_data = model.fit_transform(scaled_data)
    elif algorithm == "t-SNE":
        model = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        reduced_data = model.fit_transform(scaled_data)

    # Visualization
    st.sidebar.markdown("Step 5: Visualization")
    if algorithm in ["K-Means Clustering", "DBSCAN Clustering"]:
        data['Cluster'] = clusters

        # Display cluster summary
        st.write("Cluster Summary:")
        st.write(data.groupby('Cluster').mean())

        # Display cluster distribution
        st.write("Cluster Distribution:")
        st.bar_chart(data['Cluster'].value_counts())

        # Scatter plot of clusters
        st.write("Cluster Scatter Plot:")
        fig, ax = plt.subplots()
        sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=data['Cluster'], palette='viridis', ax=ax)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        st.pyplot(fig)

        # Additional visualization: Pair plot
        st.write("Pair Plot:")
        pair_plot = sns.pairplot(data, hue='Cluster', palette='viridis')
        st.pyplot(pair_plot)

        # Additional visualization: Elbow plot for K-Means
        st.write("Elbow Plot:")
        distortions = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=random_state)
            kmeans.fit(scaled_data)
            distortions.append(kmeans.inertia_)
        st.line_chart(distortions)

    elif algorithm in ["PCA", "FastICA", "KernelPCA", "t-SNE"]:
        if algorithm == "t-SNE":
            st.write("t-SNE Scatter Plot:")
        else:
            st.write(f"{algorithm} Scatter Plot:")

        fig, ax = plt.subplots()
        if algorithm == "t-SNE":
            sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax)
        else:
            sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis', ax=ax)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        st.pyplot(fig)

        # Additional visualization: 3D Scatter plot
        if algorithm != "t-SNE" and reduced_data.shape[1] >= 3:
            st.write("3D Scatter Plot:")
            fig_3d = plt.figure()
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=clusters, cmap='viridis')
            ax_3d.set_xlabel('Component 1')
            ax_3d.set_ylabel('Component 2')
            ax_3d.set_zlabel('Component 3')
            st.pyplot(fig_3d)
        elif algorithm != "t-SNE":
            st.write("The reduced data has fewer than 3 components, so a 3D scatter plot cannot be created.")

# Add some conclusion
st.sidebar.markdown(explanations[7])
st.write("Congratulations! You have successfully completed the analysis.")