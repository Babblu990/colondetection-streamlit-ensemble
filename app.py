import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
import os
import gdown
import time

# -------------------------------------------------
# PAGE CONFIG (Dark Medical Theme)
# -------------------------------------------------
st.set_page_config(
    page_title="Colon Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (Dark UI + Buttons)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.result-box {
    background-color: #1b5e20;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
}
.stButton>button {
    background-color: #1976d2;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🧠 Colon Disease Detection")
st.sidebar.write(
    "This system uses an **ensemble deep learning approach** "
    "to analyze colonoscopy images."
)
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ For research and educational purposes only.\n"
    "Not a medical diagnosis."
)

# -------------------------------------------------
# GOOGLE DRIVE MODEL FILE IDS
# -------------------------------------------------
VGG_ID = "1xqJGFRRQiNlyjiqzAsRhJ3zbkLmYJGUx"
RESNET_ID = "1jFE8eaxEhSxbvROZegi48FrDxDS9wiul"
INCEPTION_ID = "1for1P3876wQ48gQuuuOUKsPWfL9qv8I4"

def download_model(file_id, output_name):
    if not os.path.exists(output_name):
        st.info(f"⬇️ Downloading {output_name} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)

download_model(VGG_ID, "vgg16.h5")
download_model(RESNET_ID, "resnet50.h5")
download_model(INCEPTION_ID, "inceptionv3.h5")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    return (
        tf.keras.models.load_model("vgg16.h5"),
        tf.keras.models.load_model("resnet50.h5"),
        tf.keras.models.load_model("inceptionv3.h5"),
    )

model_vgg, model_resnet, model_inception = load_models()

# -------------------------------------------------
# CLASS LABELS
# -------------------------------------------------
class_names = ["Normal", "Polyp", "Ulcer", "Cancer"]

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.markdown("<h1 style='text-align:center;'>Colon Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>"
    "Ensemble Model (VGG16 + ResNet50 + InceptionV3)"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2 = st.columns(2)

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
with col1:
    st.subheader("📤 Upload Colonoscopy Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

# -------------------------------------------------
# PREPROCESS FUNCTION
# -------------------------------------------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        img = preprocess(image)

        if st.button("Predict"):
            with st.spinner("🔍 Analyzing image..."):
                time.sleep(1.5)

                p1 = model_vgg.predict(img)[0]
                p2 = model_resnet.predict(img)[0]
                p3 = model_inception.predict(img)[0]

                preds = [
                    class_names[np.argmax(p1)],
                    class_names[np.argmax(p2)],
                    class_names[np.argmax(p3)]
                ]

                final_prediction = Counter(preds).most_common(1)[0][0]
                avg_confidence = np.mean([
                    np.max(p1), np.max(p2), np.max(p3)
                ])

            st.markdown(
                f"<div class='result-box'>Final Prediction: {final_prediction}</div>",
                unsafe_allow_html=True
            )

            st.progress(int(avg_confidence * 100))
            st.caption(f"Confidence: {avg_confidence*100:.2f}%")

            if avg_confidence < 0.6:
                st.warning("⚠️ Low confidence prediction. Please consult a specialist.")

