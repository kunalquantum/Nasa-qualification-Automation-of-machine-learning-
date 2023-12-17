import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000
)

# Function to preprocess an image for model prediction
def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to classify an image
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions

# Function to display classification results
def display_results(predictions, top_k=5):
    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=top_k)[0]
    return decoded_predictions

# Streamlit UI
st.title("Advanced Image Classification App")

# Sidebar for uploading images
st.sidebar.title("Upload Image")

# Allow multiple file uploads
uploaded_images = st.sidebar.file_uploader("Choose one or more images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the image when the user clicks the button
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                predictions = classify_image(image)

            # Display the classification results
            st.subheader("Classification Results:")
            results = display_results(predictions)
            for i, (imagenet_id, label, score) in enumerate(results):
                st.write(f"{i + 1}: {label} ({score:.2f})")

# Information about how to use the app
st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1. Upload one or more images using the sidebar on the left.
    2. Click the "Classify Image" button to classify the uploaded image(s).
    3. The top 5 classification results will be displayed for each image.
    """
)

# Add an option to switch between different pre-trained models
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox("Choose a pre-trained model", ["ResNet50", "MobileNetV2"])

# Load the selected model
if selected_model == "ResNet50":
    model = tf.keras.applications.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000
    )
elif selected_model == "MobileNetV2":
    model = tf.keras.applications.MobileNetV2(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000
    )

st.sidebar.text(f"Selected Model: {selected_model}")