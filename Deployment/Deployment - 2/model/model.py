# model.py

import tensorflow as tf
from tensorflow import keras
from keras import layers

input_shape = (248, 248, 3)
num_classes = 3

# Define a custom attention layer with mask
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Compute attention weights
        attention_weights = tf.nn.softmax(inputs, axis=1)
        # Apply attention weights to inputs
        if mask is not None:
            attention_weights *= mask
            # Normalize attention weights
            attention_weights /= tf.reduce_sum(attention_weights, axis=1, keepdims=True)
        weighted_inputs = inputs * attention_weights
        # Sum along the feature map dimension to compute the attended features
        attended_features = tf.reduce_sum(weighted_inputs, axis=1)
        return attended_features

def attention_model():
    # Define the model architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    attention_output = AttentionLayer()(x)
    x = tf.keras.layers.Flatten()(attention_output)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[attention_output, outputs])

    return model