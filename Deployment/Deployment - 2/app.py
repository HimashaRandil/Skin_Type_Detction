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

if model is not None:

    st.title('Dermalytics - Skin Oiliness Detction App')
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
                # Provide remedies based on prediction
                if predicted_label == 'Oily':
                    st.header("Remedies for Oily Skin:")
                    st.write("Here are some tips for managing oily skin, as recommended by the Mayo Clinic (https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/what-to-do-about-dry-skin):")
                    oily_skin_remedies = [
                        "- Wash your face twice daily with a gentle cleanser.",
                        "- Use oil-free, non-comedogenic moisturizers.",
                        "- Apply blotting papers throughout the day to absorb excess oil.",
                        "- Consider using clay masks or salicylic acid products to control oil production.",
                        "- Maintain a healthy diet and drink plenty of water."
                    ]
                    for remedy in oily_skin_remedies:
                        st.write(remedy)
                elif predicted_label == 'Dry':
                    st.header("Remedies for Dry Skin:")
                    st.write("Here are some tips for managing dry skin, as recommended by the Mayo Clinic (https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/what-to-do-about-dry-skin):")
                    dry_skin_remedies = [
                        "- Use a gentle cleanser that won't strip away natural oils.",
                        "- Apply a moisturizer regularly, both in the morning and evening.",
                        "- Look for moisturizers with ingredients like hyaluronic acid, ceramides, or glycerin.",
                        "- Exfoliate your skin gently once or twice a week to remove dead skin cells.",
                        "- Use a humidifier to add moisture to the air, especially during dry winter months."
                    ]
                    for remedy in dry_skin_remedies:
                        st.write(remedy)

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
        
        
        
        if predicted_label == 'Oily':
            st.header("Remedies for Oily Skin:")
            st.write("Here are some tips for managing oily skin, as recommended by the Mayo Clinic (https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/what-to-do-about-dry-skin):")
            oily_skin_remedies = [
                "- Wash your face twice daily with a gentle cleanser.",
                "- Use oil-free, non-comedogenic moisturizers.",
                "- Apply blotting papers throughout the day to absorb excess oil.",
                "- Consider using clay masks or salicylic acid products to control oil production.",
                "- Maintain a healthy diet and drink plenty of water."
            ]
            for remedy in oily_skin_remedies:
                st.write(remedy)
        elif predicted_label == 'Dry':
            st.header("Remedies for Dry Skin:")
            st.write("Here are some tips for managing dry skin, as recommended by the Mayo Clinic (https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/what-to-do-about-dry-skin):")
            dry_skin_remedies = [
                        "- Use a gentle cleanser that won't strip away natural oils.",
                        "- Apply a moisturizer regularly, both in the morning and evening.",
                        "- Look for moisturizers with ingredients like hyaluronic acid, ceramides, or glycerin.",
                        "- Exfoliate your skin gently once or twice a week to remove dead skin cells.",
                        "- Use a humidifier to add moisture to the air, especially during dry winter months."
                    ]
            for remedy in dry_skin_remedies:
                st.write(remedy)

else:
    # Handle case where model loading fails
    st.error("Error loading the skin oiliness detection model. Please try again later.")
