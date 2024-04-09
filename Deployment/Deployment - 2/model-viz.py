from vis.visualization import model_to_dot
import streamlit as st
import tensorflow as tf
import numpy as np  # May be needed for preprocessing
import pandas as pd  # May be needed for preprocessing
from streamlit_webrtc import webrtc_streamer
from PIL import Image  # For image processing
from tensorflow import keras
from keras import layers
from keras import backend as K
from keras.losses import categorical_crossentropy

# Import the attention_model function and custom_objects from model.py
from model.model import attention_model, AttentionLayer


def load_model():
  """Loads the model, handling potential exceptions."""
  # Define custom objects (if necessary)
  custom_objects = {'AttentionLayer': AttentionLayer}  # Example

  try:
    # Load the model within custom object scope (if necessary)
    with tf.keras.utils.custom_object_scope(custom_objects):
      model = tf.keras.models.load_model('.\model\custom_model_version4_2.h5')
    return model

  except Exception as e:
    print("Error loading model:", e)
    return None  # Indicate failure

# Load the model (handle potential errors)
model = load_model()

if model is not None:  # Check if model loaded successfully
  try:
    dot = model_to_dot(model, show_shapes=True)
    Image(dot.create_png())  # Display image in IPython
  except Exception as e:
    print("Error visualizing model:", e)

# Alternative Display Method (if not using IPython):
# import matplotlib.pyplot as plt
# plt.imshow(dot.create_png())
# plt.show()