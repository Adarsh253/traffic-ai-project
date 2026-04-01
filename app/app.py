import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model/mobilenetv2_traffic.h5", compile=False)

# Class labels (you can rename later)
classes = [
    "Speed Limit 20",
    "Speed Limit 30",
    "Yield",
    "Stop",
    "No Entry"
]

st.set_page_config(page_title="Traffic AI", layout="centered")

st.title("🚦 Traffic Signal Recognition AI")
st.write("Upload an image to detect traffic sign")

# Upload image
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.convert("RGB")
    img = img.resize((96, 96))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {classes[class_id]}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.subheader("📷 Live Camera Detection")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", width=300)

    # Preprocess
    img = image.convert("RGB")
    img = img.resize((96, 96))   # ⚠️ change to 96 if using MobileNet later
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {classes[class_id]}")
    st.info(f"Confidence: {confidence*100:.2f}%")