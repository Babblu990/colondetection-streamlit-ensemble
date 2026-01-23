import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter

# -----------------------------
# Load trained models
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
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Colon Disease Detection", layout="centered")

st.title("Colon Disease Detection System")
st.subheader("Ensemble Deep Learning Model (Majority Voting)")

st.write(
    "Upload a colonoscopy image. The system internally uses multiple deep learning models "
    "and provides a single final prediction."
)

uploaded_file = st.file_uploader(
    "Upload colonoscopy image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction logic
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)

    if st.button("Predict"):
        # Predictions from all models
        pred_vgg = np.argmax(model_vgg.predict(img))
        pred_resnet = np.argmax(model_resnet.predict(img))
        pred_inception = np.argmax(model_inception.predict(img))

        # Convert predictions to labels
        predictions = [
            class_names[pred_vgg],
            class_names[pred_resnet],
            class_names[pred_inception]
        ]

        # Majority voting
        final_prediction = Counter(predictions).most_common(1)[0][0]

        # Display ONLY final prediction
        st.success(f"Final Prediction: {final_prediction}")

        # Medical disclaimer
        st.caption(
            "⚠️ This result is for research and educational purposes only and "
            "should not be considered a medical diagnosis."
        )
