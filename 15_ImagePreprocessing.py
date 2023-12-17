import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Helper function to process the image
def process_image(image, options):
    # Resize the image
    resized_image = cv2.resize(image, (options["width"], options["height"]))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    return gray_image

# Helper function for object detection using SSD MobileNet V2
def perform_object_detection(image):
    # Load the pre-trained SSD MobileNet V2 model
    model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")

    # Load label map
    category_index = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'}}

    # Convert the image to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(image)

    # Perform object detection
    detections = model(input_tensor)

    return detections, category_index

# Home page
def home():
    st.title("Image Processing ")
    st.write("Welcome to the Image Processing ")
    st.write("Please select an option from the sidebar.")

# Image Processing and Object Detection page
def image_processing_and_detection():
    st.title("Image Processing ")

    # Upload Image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Pre-processing Options
    st.sidebar.title("Pre-processing Options")
    width = st.sidebar.slider("Width", 100, 1000, 500)
    height = st.sidebar.slider("Height", 100, 1000, 500)
    options = {"width": width, "height": height}

    # Process Image Button
    if st.sidebar.button("Process Image") and uploaded_image is not None:
        # Load the uploaded image
        img = Image.open(uploaded_image)
        img_array = np.array(img)

        # Pre-process the image
        processed_image = process_image(img_array, options)

        # Display the processed image
        st.subheader("Processed Image")
        st.image(processed_image, caption="Processed Image", use_column_width=True)


# Main app
def main():
    st.set_page_config(page_title="Image App", layout="wide")

    # Create a sidebar menu
    st.sidebar.title("Navigation")
    pages = {
        "Home": home,
        "Image Processing ": image_processing_and_detection,
    }
    selected_page = st.sidebar.radio("", list(pages.keys()))

    # Display the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
