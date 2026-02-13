import streamlit as st
from streamlit.components.v1 import html
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
import os
import gdown
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Colon Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM HTML + CSS (FRONTEND)
# -------------------------------------------------
frontend_html = """
<style>
body {
    background-color: #f4f6f9;
    font-family: Arial, sans-serif;
}
.card {
    background: white;
    width: 500px;
    margin: auto;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
}
.title {
    font-size: 26px;
    font-weight: bold;
    color: #0d47a1;
}
.subtitle {
    color: #555;
    margin-bottom: 20px;
}
.note {
    font-size: 13px;
    color: #777;
}
</style>

<div class="card">
    <div class="title">Colon Disease Detection</div>
    <div class="subtitle">
        AI-based Colonoscopy Image Analysis
    </div>
    <p class="note">
        Upload an image below and click Predict
    </p>
</div>
"""

html(frontend_html, height=320)

# -------------------------------------------------
# GOOGLE DRIVE MODEL FILE IDS
# -------------------------------------------------
VGG_ID = "1xqJGFRRQiNlyjiqzAsRhJ3zbkLmYJGUx"
RESNET_ID = "1jFE8eaxEhSxbvROZegi48FrDxDS9wiul"
INCEPTION_ID = "1for1P3876wQ48gQuuuOUKsPWfL9qv8I4"

# -------------------------------------------------
# DOWNLOAD MODELS
# -------------------------------------------------
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
        tf.keras.models.load_model("inceptionv3.h5")
    )

model_vgg, model_resnet, model_inception = load_models()

# -------------------------------------------------
# CLASS LABELS
# -------------------------------------------------
class_names = ["Normal", "Polyp", "Ulcer", "Cancer"]

# -------------------------------------------------
# STREAMLIT INTERACTION (REAL BACKEND)
# -------------------------------------------------
st.markdown("---")

st.subheader("📤 Upload Colonoscopy Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            time.sleep(1)

            img = preprocess(image)

            p1 = model_vgg.predict(img)[0]
            p2 = model_resnet.predict(img)[0]
            p3 = model_inception.predict(img)[0]

            preds = [
                class_names[np.argmax(p1)],
                class_names[np.argmax(p2)],
                class_names[np.argmax(p3)]
            ]

            final_prediction = Counter(preds).most_common(1)[0][0]
            confidence = np.mean([np.max(p1), np.max(p2), np.max(p3)])

        st.success(f"✅ Final Prediction: {final_prediction}")
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence*100:.2f}%")

        if confidence < 0.6:
            st.warning("⚠️ Low confidence prediction. Please consult a specialist.")

st.markdown("---")
st.caption(
    "⚠️ This system is for research and educational purposes only. "
    "It is not a medical diagnosis."
)
