import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import backend as K
from model.model import attention_model, AttentionLayer

# Function to build and load the model
def build_model():
    custom_objects = {'AttentionLayer': AttentionLayer}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = attention_model()
    return model

# Load the model
model = build_model()

# Function to preprocess the input image
def preprocess_image(image_path, input_shape):
     # Load the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to match the input shape
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    # Convert the image to float32 and normalize
    img_array = img.astype(np.float32) / 255.0
    # Expand the dimensions for batch processing
    img_tensor = tf.expand_dims(img_array, axis=0)
    return img_tensor, img

# Function to generate saliency map
def generate_saliency_map(model, img_tensor, original_image):
      with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)[0]  # Access first element for single image
      grads = tape.gradient(prediction, img_tensor)[0]
      grads = tf.maximum(grads, 0) / (tf.reduce_mean(grads) + tf.keras.backend.epsilon())

     # Convert grads to NumPy array for OpenCV
      grads_np = grads.numpy()

    # Resize using OpenCV
      saliency_map = cv2.resize(grads_np, (original_image.shape[1], original_image.shape[0]))

    # Normalize for better visualization (optional)
      saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

      return saliency_map

# Define input shape
input_shape = (248, 248, 3)

# Path to the image
image_path = 'assets\Oily (8).jpg'

# Preprocess the input image
img_tensor, original_image = preprocess_image(image_path, input_shape)

# Generate saliency map
saliency_map = generate_saliency_map(model, img_tensor, original_image)

# Display the saliency map


alpha = 0.5  # Adjust transparency for better visibility
saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)  # Convert to uint8 for colormap
original_image_uint8 = original_image.astype(np.uint8)  # Convert original image to uint8
overlay = cv2.addWeighted(original_image_uint8, 1 - alpha, cv2.applyColorMap(saliency_map_uint8, cv2.COLORMAP_HOT), alpha, 0)
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlayed Saliency Map')
plt.show()