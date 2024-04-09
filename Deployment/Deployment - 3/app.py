import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 
from streamlit_webrtc import webrtc_streamer

# Assuming model.py is in the same directory
from model.model import ChannelAttentionLayer, AttentionLayer

MODEL_FILE_PATH = 'model\custom_model_version_7.h5'

class_names = ['Dry', 'Normal', 'Oily']

input_shape = (248, 248, 3)
num_classes = 3

def load_model():
    model = tf.keras.models.Sequential([
    # Convolutional layers
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    
    # Hybrid Attention
    ChannelAttentionLayer(), # Apply channel attention first
    AttentionLayer(),  

    # Flatten and Dense layers for classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

    # Load the weights from the .h5 file
    model.load_weights(MODEL_FILE_PATH) 

    return model

# Function to load and preprocess image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((248, 248))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image(image):
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

# Streamlit app
def main():
  model = load_model()
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
                  class_index = np.argmax(prediction)

                  # Get predicted class name and confidence
                  predicted_class = class_names[class_index]
                  confidence = prediction[0][class_index] * 100

                  # Display results
                  st.write(f"**Predicted Skin Type: {predicted_class}**")
                  st.write(f"**Confidence: {confidence:.2f}%**")
                  
                  # Provide remedies based on prediction
                  if predicted_class == 'Oily':
                      st.header("Remedies for Oily Skin:")
                      st.write("Here are some tips for managing oily skin, as recommended by the AAD (https://www.aad.org/public/everyday-care/skin-care-basics/dry/oily-skin):")
                      oily_skin_remedies = [
                          "- Wash your face twice daily with a gentle cleanser.",
                          "- Use oil-free, non-comedogenic moisturizers.",
                          "- Apply blotting papers throughout the day to absorb excess oil.",
                          "- Consider using clay masks or salicylic acid products to control oil production.",
                          "- Maintain a healthy diet and drink plenty of water."
                      ]
                      for remedy in oily_skin_remedies:
                          st.write(remedy)
                  elif predicted_class == 'Dry':
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
          class_index = np.argmax(prediction)

          # Get predicted class name and confidence
          predicted_class = class_names[class_index]
          confidence = prediction[0][class_index] * 100

          # Display results
          st.write(f"**Predicted Skin Type: {predicted_class}**")
          st.write(f"**Confidence: {confidence:.2f}%**")
          
          
          
          if predicted_class == 'Oily':
              st.header("Remedies for Oily Skin:")
              st.write("Here are some tips for managing oily skin, as recommended by the AAD (https://www.aad.org/public/everyday-care/skin-care-basics/dry/oily-skin):")
              oily_skin_remedies = [
                  "- Wash your face twice daily with a gentle cleanser.",
                  "- Use oil-free, non-comedogenic moisturizers.",
                  "- Apply blotting papers throughout the day to absorb excess oil.",
                  "- Consider using clay masks or salicylic acid products to control oil production.",
                  "- Maintain a healthy diet and drink plenty of water."
              ]
              for remedy in oily_skin_remedies:
                  st.write(remedy)
          elif predicted_class == 'Dry':
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
    

if __name__ == "__main__":
    main()