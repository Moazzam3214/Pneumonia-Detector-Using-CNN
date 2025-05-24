import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2

# Load model
model = load_model('pneumonia_detection_model.h5')

st.title("Pneumonia Detection")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize image for model
    img_resized = cv2.resize(img, (150, 150))
    img_input = img_resized.reshape(1, 150, 150, 1) / 255.0

    # Predict
    pred = model.predict(img_input)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred

    # Display prediction and image side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            f"<h1 style='color: {'red' if label == 'Pneumonia' else 'green'};'>{label}</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3>Confidence: {confidence:.2f}</h3>", unsafe_allow_html=True)

    with col2:
        st.image(img, caption="Uploaded Image", width=300)
