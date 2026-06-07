# CrdioSense AI - phase 3 : streamlit web APP
# Author: prince-gupta79


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# load mode; and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# page config
st.set_page_config(
    page_title = 'cardiosense AI',
    page_icon = "🤓🦉",
    layout = "centered"
)

st.title("CadrioSense AI")
st.subheader("Heart Disease Prediction System")
st.markdown("Enter Patient deatils below to get a prediction")
st.divider()

# input form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=54)
    trestbps = st.number_input("Blood Pressure (mm HG)", min_value=80, max_value=220, value=120)
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], index = 1)
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4], index = 0)


with col2:
     sex = st.selectbox("Sex", [0, 1], index = 1, format_func = lambda x: "Female" if x == 0 else "Male")
     chol = st.number_input("Cholestrol (mg/dl)", min_value = 100, max_value = 600, value = 246)
     thalach = st.number_input("Max Heart Rate", min_value = 60, max_value = 220, value = 150)
     thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])


with col3:
        cp = st.selectbox("Chest Pain Type(cp)", [0, 1, 2, 3], index = 0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index = 0, format_func = lambda x: "True " if x == 1 else"False")
        slope = st.selectbox("ST Slope", [0, 1, 2], index = 1)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], index = 0, format_func =lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (old peak)", min_value = 0.0, max_value = 7.0, value = 1.0 )

        st.divider()

if st.button("Analyse Patient", use_container_width = True):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal
                            ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Clinical Assessment Results")

    if prediction == 1:
        st.error(f"HIGH RISK - Heart diesase Detected")
        st.metric("probability", f"{probability * 100:.1f}%")
        st.warning("Recommendation: Immediate cardiology consultation advised")

    else:
        st.success(f"LOW RICK - No Heart Disease Detected")
        st.metric("Probability", f"{probability *100:.1f}%")
        st.info("Recommendation: Continue regular health monitoring")

    st.divider()
    st.caption("This tool is for educational purposes only. Not a substitute for professional medical adive...")