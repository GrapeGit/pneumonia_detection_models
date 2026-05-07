import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap( model: tf.keras.Model, image_array: np.ndarray, last_conv_layer_name: str | None = None):

    """Generate a Grad-CAM heatmap for one preprocessed image."""


    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        pneumonia_score = predictions[:, 0]

    grads = tape.gradient(pneumonia_score, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(
    original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4, ):
    """
    Overlay a red Grad-CAM heatmap on the original image.
    """

    original_image = original_image.convert("RGB")
    original_size  = original_image.size

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(original_size)
    heatmap_img = heatmap_img.convert("L")

    red_overlay = Image.new("RGB", original_size, color=(255, 0, 0))
    mask = heatmap_img.point(lambda p: int(p * alpha))

    overlayed = Image.composite(red_overlay, original_image, mask)
    return overlayed