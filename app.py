import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap


MODEL_PATH = "cnn_section2_best.keras"
IMAGE_SIZE = (64, 64)
THRESHOLD = 0.95


@st.cache_resource
def load_model():
    """Load the saved Keras model once and cache it for Streamlit."""
    return tf.keras.models.load_model(MODEL_PATH)


def process_image(image: Image.Image) -> np.ndarray:
    """Convert, resize, normalize, and batch an uploaded image."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict(model, image_array: np.ndarray) -> float:
    """Return the model's predicted pneumonia probability."""
    prediction = model.predict(image_array, verbose=0)
    return float(prediction[0][0])


def show_prediction(probability: float):
    """Display the prediction result in a clean UI block."""
    diagnosis = "Pneumonia" if probability >= THRESHOLD else "No Pneumonia"

    st.subheader("Prediction Result")

    if diagnosis == "Pneumonia":
        st.error("Prediction: Pneumonia")
    else:
        st.success("Prediction: No Pneumonia")

    st.metric(
        label="Pneumonia Probability",
        value=f"{probability * 100:.1f}%",
    )

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("How this prediction is made"):
        st.write(
            f"The model outputs a probability between 0 and 1. "
            f"This app uses a decision threshold of **{THRESHOLD:.2f}**. "
            f"If the probability is greater than or equal to this threshold, "
            f"the image is classified as **Pneumonia**."
        )


def main():
    st.set_page_config(
        page_title="Pneumonia Detection",
        page_icon="🫁",
        layout="centered",
    )

    st.title("Pneumonia Detection CNN")
    st.write(
        "Upload a chest X-ray image and the model will predict whether it shows "
        "**Pneumonia** or **No Pneumonia**."
    )

    st.warning(
        "This app is for research and educational purposes only. "
        "It is not intended to provide medical advice, diagnosis, or treatment."
    )

    with st.sidebar:
        st.header("Model Info")
        st.write("**Model:** CNN")
        st.write("**Input size:** 64 × 64 × 3")
        st.write(f"**Decision threshold:** {THRESHOLD}")
        st.write("**Test accuracy:** 83.3%")
        st.write("**Field accuracy:** 64.0%")

        st.divider()

        st.caption(
            "The model was trained with TensorFlow/Keras and uses a sigmoid "
            "output for binary classification."
        )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    image = Image.open(uploaded_file)
    processed_image = process_image(image)
    probability = predict(model, processed_image)

    heatmap = make_gradcam_heatmap(model, processed_image)
    heatmap_overlay = overlay_heatmap(image, heatmap)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Chest X-ray image",
            use_container_width=True,
        )

    with col2:
        show_prediction(probability)

    st.divider()

    st.caption(
        "Note: Model predictions can be incorrect, especially on images that differ "
        "from the training data. The field-set accuracy was lower than the test-set "
        "accuracy, so results should be interpreted cautiously."
    )

    st.divider()

    st.subheader("Grad-CAM Heatmap")
    st.write(
        "The heatmap highlights regions that most influenced the model's prediction. "
        "It should be used only as an interpretability aid."
    )

    st.image(
        heatmap_overlay,
        caption="Grad-CAM heatmap overlay",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()