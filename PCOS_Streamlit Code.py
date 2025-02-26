#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np

model_path = "C:\\Users\\Vanshika Rana\\Downloads\\pcos_rf_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("PCOS Prediction App")
st.write("Enter the patient details below to predict the likelihood of PCOS.")

height = st.number_input("Height (cm)", min_value=120.0, max_value=200.0, value=160.0)
weight = st.number_input("Weight (Kg)", min_value=30.0, max_value=150.0, value=55.0)
follicle_l = st.number_input("Follicle No. (Left)", min_value=0, max_value=30, value=5)
follicle_r = st.number_input("Follicle No. (Right)", min_value=0, max_value=30, value=5)
avg_fsize_r = st.number_input("Avg. Follicle Size (Right) (mm)", min_value=5.0, max_value=30.0, value=15.0)
pimples = st.radio("Do you have Pimples?", ["No", "Yes"])
pimples = 1 if pimples == "Yes" else 0

input_data = np.array([[avg_fsize_r, follicle_l, pimples, height, weight]])

# Predict button
if st.button("Predict PCOS"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("The model predicts **PCOS detected**.")
    else:
        st.success("The model predicts **No PCOS detected**.")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF5F5;
    }
    .stButton>button {
        background-color: #FF4081 !important;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #E91E63 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[ ]:




