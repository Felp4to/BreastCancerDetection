import cv2
import os
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def normalize_image(image_path):
    img = load_img(image_path, target_size=(50, 50))
    img = img_to_array(img)
    img = img / 255.0
    input_image = tf.expand_dims(img, axis=0) 

    return input_image


def compute_gradcam(model, folder, preprocessing_method, class_index=None):
    heatmaps = []

    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    for image_path in image_paths:
        # open and normalize image
        img = normalize_image(image_path)
        
        # define grad model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer("last_conv_layer").output, model.output]
        )

        # predictions
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            if class_index is None:
                class_index = np.argmax(predictions[0])
            output = predictions[:, class_index]

        # calculate the gradient with respect to the output
        grads = tape.gradient(output, conv_outputs)

        # take the gradient average on the channels
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # product gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

        heatmaps.append((img, heatmap))

    if preprocessing_method:
        plot_heatmaps(heatmaps, preprocessing_method)

    return heatmaps


def plot_heatmaps(heatmaps, preprocessing_method):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (img, heatmap) in enumerate(heatmaps):
        ax = axes[i // 4, i % 4]
        ax.imshow(img[0])
        ax.imshow(heatmap, cmap='jet', alpha=0.3)
        ax.axis('off')
    
    plt.tight_layout()
    save_location = os.path.join('..', 'models', preprocessing_method, 'gradCAM.jpg')
    plt.savefig(save_location, bbox_inches='tight', pad_inches=0)
    plt.show()