import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
import os
import gdown

st.set_page_config(page_title="Colon Disease Detection", layout="centered")

# -----------------------------
# Google Drive File IDs (YOUR MODELS)
# -----------------------------
VGG_ID = "1xqJGFRRQiNlyjiqzAsRhJ3zbkLmYJGUx"
RESNET_ID = "1jFE8eaxEhSxbvROZegi48FrDxDS9wiul"
INCEPTION_ID = "1for1P3876wQ48gQuuuOUKsPWfL9qv8I4"

# Local file names (must match loading)
VGG_FILE = "vgg16.h5"
RESNET_FILE = "resnet50.h5"
INCEPTION_FILE = "inceptionv3.h5"


# -----------------------------
# Download model from Google Drive
# -----------------------------
def download_model(file_id, output_name):
    if not os.path.exists(output_name):
        st.info(f"Downloading model: {output_name} ... Please wait ⏳")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)


# Download models if missing
download_model(VGG_ID, VGG_FILE)
download_model(RESNET_ID, RESNET_FILE)
download_model(INCEPTION_ID, INCEPTION_FILE)


# -----------------------------
# Load Models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    model_vgg = tf.keras.models.load_model(VGG_FILE)
    model_resnet = tf.keras.models.load_model(RESNET_FILE)
    model_inception = tf.keras.models.load_model(INCEPTION_FILE)
    return model_vgg, model_resnet, model_inception


model_vgg, model_resnet, model_inception = load_models()

# -----------------------------
# Class Labels (CHANGE IF YOUR CLASSES DIFFER)
# -----------------------------
class_names = ["Normal", "Polyp", "Ulcer", "Cancer"]

# -----------------------------
# UI
# -----------------------------
st.title("Colon Disease Detection System")
st.subheader("Ensemble Deep Learning (Majority Voting)")

st.write("Upload a colonoscopy image and get the final prediction.")

uploaded_file = st.file_uploader(
    "Upload Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)

    if st.button("Predict"):
        try:
            # Predict from all three models
            pred_vgg = np.argmax(model_vgg.predict(img))
            pred_resnet = np.argmax(model_resnet.predict(img))
            pred_inception = np.argmax(model_inception.predict(img))

            # Convert to labels
            results = [
                class_names[pred_vgg],
                class_names[pred_resnet],
                class_names[pred_inception],
            ]

            # Majority Voting
            final_prediction = Counter(results).most_common(1)[0][0]

            # Display only FINAL output
            st.success(f"Final Prediction: {final_prediction}")

            st.caption("⚠️ For research and educational purposes only. Not a medical diagnosis.")

        except Exception as e:
            st.error("Something went wrong while predicting.")
            st.code(str(e))
