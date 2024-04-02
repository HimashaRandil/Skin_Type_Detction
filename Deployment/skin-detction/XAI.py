import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

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


# Define input shape and number of classes
input_shape = (248, 248, 3)
num_classes = 3

# Define a function to preprocess the image
def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    # Resize the image to match the input shape
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    # Convert the image to float32 and normalize
    img_array = img.astype(np.float32) / 255.0
    return img_array

# Define a function to generate Grad-CAM heatmap
def generate_grad_cam(model, img_array):
    # Convert the image array to a tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Expand the dimensions for batch processing
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # Get the attention output from the model
    attention_output = model(img_tensor)

    # Check if the attention output is a list or a tensor
    if isinstance(attention_output, list):
        attention_output = attention_output[0]

    # Get the target layer output
    target_layer_output = attention_output

    # Compute the gradient of the model output with respect to the attention output
    with tf.GradientTape() as tape:
        tape.watch(attention_output)
        loss = tf.reduce_mean(attention_output)

    # Compute the gradients
    grads_target = tape.gradient(loss, attention_output)

    if grads_target is None:
        print("Gradients are None")
        print("Attention output shape:", attention_output.shape)
        print("Target layer output shape:", target_layer_output.shape)
        raise ValueError("Gradients are None")

    # Compute the weighted sum of the target layer output using the gradients directly
    cam_target = tf.reduce_sum(grads_target, axis=-1)

    # Apply ReLU to ensure only positive values contribute to the heatmap
    cam_target = tf.nn.relu(cam_target)

    # Normalize the heatmap
    cam_target /= tf.reduce_max(cam_target)

    # Ensure cam_target has 4 dimensions
    cam_target = tf.expand_dims(cam_target, axis=-1)
    cam_target = tf.expand_dims(cam_target, axis=0)

    # Resize the heatmap to match the image size
    heatmap_target = tf.image.resize(cam_target, (img_array.shape[0], img_array.shape[1]))

    # Convert the heatmap to a NumPy array
    heatmap_target = heatmap_target.numpy()

    return heatmap_target






# Path to the image
image_path = 'assets\dry 17.jpg'

# Preprocess the image
img_array = preprocess_image(image_path)

# Generate Grad-CAM heatmap
heatmap_target = generate_grad_cam(model, img_array)

# Resize the heatmap to match the image size
heatmap_target = cv2.resize(heatmap_target, (img_array.shape[1], img_array.shape[0]))

# Apply colormap to the heatmap
heatmap_target = cv2.applyColorMap(np.uint8(255 * heatmap_target), cv2.COLORMAP_JET)

# Combine the heatmap with the original image
grad_cam_target = cv2.addWeighted(img_array.astype(np.float32), 0.8, heatmap_target.astype(np.float32), 0.4, 0)


# Display the Grad-CAM visualization
plt.figure(figsize=(10, 5))
plt.imshow(grad_cam_target)
plt.title('Grad-CAM (Target)')
plt.axis('off')
plt.show()