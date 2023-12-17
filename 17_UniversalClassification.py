import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Function to fetch and preprocess images
def fetch_and_preprocess_images(data_dir, image_exts):
    image_data = []
    labels = []
    class_labels = []  # Store class labels
    
    for image_class in os.listdir(data_dir):
        class_labels.append(image_class)  # Add class label
        
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip in image_exts:
                    img = cv2.resize(img, (256, 256))  # Resize images
                    img = img / 255.0  # Normalize pixel values
                    image_data.append(img)
                    labels.append(class_labels.index(image_class))  # Assign labels
            except Exception as e:
                print('Issue with image {}'.format(image_path))

    return np.array(image_data), np.array(labels), class_labels

# Function to create a customizable model
def create_custom_model(layer_configs):
    model = Sequential()
    for config in layer_configs:
        if config["layer_type"] == "Conv2D":
            model.add(Conv2D(config["filters"], config["kernel_size"], strides=config["strides"], activation=config["activation"]))
        elif config["layer_type"] == "MaxPooling2D":
            model.add(MaxPooling2D(pool_size=config["pool_size"]))
        elif config["layer_type"] == "Dense":
            model.add(Dense(config["units"], activation=config["activation"]))
        elif config["layer_type"] == "Dropout":
            model.add(Dropout(config["rate"]))
    return model

# Streamlit app for image classification
st.title("Image Classification App")
st.write("Upload an image and let the model classify it.")

# User-defined hyperparameters and layers
num_epochs = st.slider("Number of Epochs", 1, 50, 20)
layer_configs = []  # Store user-defined layer configurations

st.subheader("Add Layers to the Model")
add_layer = st.button("Add Layer")
while add_layer:
    layer_type = st.selectbox("Layer Type", ["Conv2D", "MaxPooling2D", "Dense", "Dropout"])
    if layer_type == "Conv2D":
        filters = st.number_input("Filters", 1, 256, 16)
        kernel_size = st.slider("Kernel Size", 1, 10, 3)
        strides = st.slider("Strides", 1, 10, 1)
        activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])
        layer_configs.append({"layer_type": "Conv2D", "filters": filters, "kernel_size": (kernel_size, kernel_size), "strides": (strides, strides), "activation": activation})
    elif layer_type == "MaxPooling2D":
        pool_size = st.slider("Pool Size", 1, 10, 2)
        layer_configs.append({"layer_type": "MaxPooling2D", "pool_size": (pool_size, pool_size)})
    elif layer_type == "Dense":
        units = st.number_input("Units", 1, 1024, 256)
        activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])
        layer_configs.append({"layer_type": "Dense", "units": units, "activation": activation})
    elif layer_type == "Dropout":
        rate = st.slider("Dropout Rate", 0.0, 1.0, 0.5)
        layer_configs.append({"layer_type": "Dropout", "rate": rate})
    
    add_layer = st.button("Add Another Layer")

# Load and preprocess image data
data_dir = 'D:\\Space\\multipleapp_pages\\data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
image_data, labels, class_labels = fetch_and_preprocess_images(data_dir, image_exts)

# Split the data into train, validation, and test sets
train_size = int(len(image_data) * 0.7)
val_size = int(len(image_data) * 0.2)
test_size = len(image_data) - train_size - val_size

x_train, x_val, x_test = (
    image_data[:train_size],
    image_data[train_size:train_size + val_size],
    image_data[train_size + val_size:],
)
y_train, y_val, y_test = (
    labels[:train_size],
    labels[train_size:train_size + val_size],
    labels[train_size + val_size:],
)

# Training the model
if st.button("Train Model"):
    st.write("Training the model...")
    model = create_custom_model(layer_configs)
    model.add(Flatten())
    model.add(Dense(len(class_labels), activation='softmax'))  # Softmax for multi-class
    model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    # Callback to display training progress in Streamlit
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.write(f"Epoch {epoch + 1}/{num_epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")
    
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, callbacks=[TrainingProgressCallback()])
    
    # Save the trained model
    model.save('imageclassifier.h5')
    st.write("Training completed!")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    # Read and preprocess the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0

    # Load the saved model
    saved_model = load_model('imageclassifier.h5')

    # Make a prediction
    prediction = saved_model.predict(np.expand_dims(image, 0))

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display the prediction
    predicted_class_index = np.argmax(prediction)
    st.write("Predicted class is:", class_labels[predicted_class_index])