import streamlit as st
import tensorflow as tf
import numpy as np  # May be needed for preprocessing
import pandas as pd  # May be needed for preprocessing
from streamlit_webrtc import webrtc_streamer
from PIL import Image  # For image processing
from tensorflow import keras
from keras import layers

# Import the attention_model function and custom_objects from model.py
from model.model import attention_model, AttentionLayer

# Load the model within custom object scope
def load_model():
    # Define custom objects
    custom_objects = {'AttentionLayer': AttentionLayer}
    
    # Load the model within custom object scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Load the model
        model = tf.keras.models.load_model('.\model\custom_model_version4_2.h5')

    return model

# Load the model
model = load_model()
# Load the pre-trained model with custom objects argument

def preprocess_image(image):
    """Preprocesses an image for model input.

    Args:
        image: A PIL Image object.

    Returns:
        A NumPy array representing the preprocessed image.
    """

    # Convert the image to RGB format
    image = image.convert('RGB')

    # Resize the image to a fixed size
    image = image.resize((248, 248))  # Adjust based on your model's input shape

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize pixel values (assuming your model expects values between 0 and 1)
    image_array = image_array / 255.0

    # Expand the dimension for batch processing
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


st.title('Skin Oiliness Classification App')
st.write('Choose how you want to provide a skin image:')

# Option 1: Capture from Webcam
if st.button('Capture from Webcam'):
    ctx = webrtc_streamer(key="capture")
    if ctx.video_capture:
        frame = ctx.video_capture.get_frame()
        if frame is not None:
            # Convert frame to PIL Image for processing
            pil_image = Image.fromarray(frame)

            # Preprocess the image
            preprocessed_image = preprocess_image(pil_image)

            # Make prediction using the loaded model
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction)

            # Map the class index to the corresponding class label
            class_labels = ['Dry','Normal','Oily']   
            predicted_label = class_labels[predicted_class]

            # Display the prediction result
            st.success(f"Predicted Skin Type: {predicted_label}")

# Option 2: Upload an image
uploaded_image = st.file_uploader("Upload an image of your skin:", type=['jpg', 'jpeg', 'png'])
if uploaded_image is not None:
    # Read the uploaded image as a PIL Image
    pil_image = Image.open(uploaded_image)

    # Preprocess the image
    preprocessed_image = preprocess_image(pil_image)

    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Map the class index to the corresponding class label
    class_labels = ['Dry','Normal','Oily']  
    predicted_label = class_labels[predicted_class]

    # Display the prediction result
    st.success(f"Predicted Skin Type: {predicted_label}")
