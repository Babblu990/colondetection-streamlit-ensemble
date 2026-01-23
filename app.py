import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
import os
import gdown

st.set_page_config(page_title="Colon Disease Detection", layout="centered")

# -----------------------------
# Google Drive direct download links
# -----------------------------
VGG_URL = "https://drive.google.com/uc?export=download&id=1xqJGFRRQiNlyjiqzAsRhJ3zbkLmYJGUx"
RESNET_URL = "https://drive.google.com/uc?export=download&id=1jFE8eaxEhSxbvROZegi48FrDxDS9wiul"
INCEPTION_URL = "https://drive.google.com/uc?export=download&id=1for1P3876wQ48gQuuuOUKsPWfL9qv8I4"

# -----------------------------
# Download models if not present
# -----------------------------
def download_model(url, output_name):
    if not os.path.exists(output_name):
        st.info(f"Downloading model: {output_name} ... Please wait ⏳")
        gdown.download(url, output_name, quiet=False)

download_model(VGG_URL, "vgg16.h5")
download_model(RESNET_URL, "resnet50.h5")
download_model(INCEPTION_URL, "inceptionv3.h5")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    model_vgg = tf.keras.models.load_model("vgg16.h5")
    model_resnet = tf.keras.models.load_model("resnet50.h5")
    model_inception = tf.keras.models.load_model("inceptionv3.h5")
    return model_vgg, model_resnet, model_inception

model_vgg, model_resnet, model_inception = load_models()

# -----------------------------
# Class labels (must match training)
# -----------------------------
class_names = ["Normal", "Polyp", "Ulcer", "Cancer"]

# -----------------------------
# UI
# -----------------------------
st.title("Colon Disease Detection System")
st.subheader("Ensemble Model (Majority Voting)")

st.write("Upload a colonoscopy image to get the final prediction.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
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
        pred_vgg = np.argmax(model_vgg.predict(img))
        pred_resnet = np.argmax(model_resnet.predict(img))
        pred_inception = np.argmax(model_inception.predict(img))

        results = [
            class_names[pred_vgg],
            class_names[pred_resnet],
            class_names[pred_inception]
        ]

        # Majority Voting Final Output
        final_prediction = Counter(results).most_common(1)[0][0]

        st.success(f"Final Prediction: {final_prediction}")

        st.caption("⚠️ This result is for research and educational purposes only.")
