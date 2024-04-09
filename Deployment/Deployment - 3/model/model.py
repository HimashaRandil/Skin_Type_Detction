import tensorflow as tf

input_shape = (248, 248, 3)
num_classes = 3

class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttentionLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_layer_one = tf.keras.layers.Dense(channels // self.ratio, activation='relu', use_bias=True)
        self.shared_layer_two = tf.keras.layers.Dense(channels, use_bias=True)
        super(ChannelAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Squeeze spatial dimensions
        squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        
        # Shared MLP
        excitation = self.shared_layer_one(squeeze)
        excitation = self.shared_layer_two(excitation)

        # Channel attention weights
        attention = tf.nn.sigmoid(excitation)
        
        # Scale input by attention weights
        outputs = inputs * attention
        return outputs
    
    def get_config(self):
        config = super(ChannelAttentionLayer, self).get_config()
        config.update({
            'ratio': self.ratio,
        })
        return config

# Define a spatial attention layer with mask
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
        # Apply attention weights to inputs
        weighted_inputs = inputs * attention_weights
        # Sum along the feature map dimension to compute the attended features
        attended_features = tf.reduce_sum(weighted_inputs, axis=1)
        return attended_features

# Define input shape and number of classes


# Create a Sequential model
