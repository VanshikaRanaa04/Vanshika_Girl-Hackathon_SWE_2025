import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("C:\\Users\\Vanshika Rana\\Downloads\\pneumonia_classifier_model.h5")

IMG_SIZE = (180, 180)

def preprocess_image(image):
    image = image.convert("RGB")  
    image = image.resize(IMG_SIZE)  
    image = np.array(image, dtype=np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

page_bg = """
<style>
    .stApp {
        background-color: #FFF5F5;
    }
    .title {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        color: #333;
    }
    .stButton>button {
        background-color: #FF4081 !important;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 18px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    .result {
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 class='title'>Pneumonia Detection</h1>", unsafe_allow_html=True)
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Predict"):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]

        st.markdown("<h3 class='result'>Prediction Result:</h3>", unsafe_allow_html=True)
        if prediction ==1:
            st.error("Pneumonia Detected!")
        else:
            st.success("No Pneumonia Detected!")



