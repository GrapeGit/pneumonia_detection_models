import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = "cnn_section2_best.keras"
IMAGE_SIZE = (64, 64)
THRESHOLD = 0.5


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def process_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict(model, image_array: np.ndarray) -> float:
    prediction = model.predict(image_array, verbose=0)
    return float(prediction[0][0])


def main():
    st.set_page_config(
        page_title="Pneumonia Detection",
        page_icon="🫁",
        layout="centered",
    )

    st.title("Pneumonia Detection")
    st.write("Upload a chest X-ray image to get a model prediction.")

    st.caption("This app is for research purposes only and not intended as medical advice.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    processed_image = process_image(image)

    probability = predict(model, processed_image)
    diagnosis = "Pneumonia" if probability >= THRESHOLD else "No Pneumonia"

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Prediction")
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Pneumonia probability:** {probability:.3f}")


if __name__ == "__main__":
    main()