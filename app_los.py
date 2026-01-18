def app():
    import streamlit as st
    # --- EXISTING CODE BELOW (UNCHANGED) ---
    st.title("Length of Stay Prediction")
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# Load trained pipeline
# ----------------------------------
pipeline = joblib.load(r"D:\Final_Project\LOS\los_final.pkl")

# ----------------------------------
# UI
# ----------------------------------
st.set_page_config(page_title="LOS Prediction", layout="centered")
st.title("🏥 Hospital Length of Stay Prediction")

# ----------------------------------
# Inputs
# ----------------------------------
age = st.slider("Age", 18, 90, 45)
gender = st.selectbox("Gender", ["Male", "Female"])
severity = st.selectbox("Severity (1–5)", [1, 2, 3, 4, 5])
comorbidities = st.selectbox("Comorbidities", [0, 1, 2, 3, 4, 5])
procedure_code = st.selectbox("Procedure Code", [1, 2, 3, 4, 5, 6])
diagnosis_code = st.selectbox("Diagnosis Code", ["D1", "D2", "D3", "D4"])
admission_type = st.selectbox(
    "Admission Type",
    ["Emergency", "Elective", "Urgent"]
)

# ----------------------------------
# Input DataFrame (STRICT)
# ----------------------------------
input_df = pd.DataFrame({
    "age": [int(age)],
    "gender": [gender],
    "severity": [int(severity)],
    "comorbidities": [int(comorbidities)],
    "procedure_code": [int(procedure_code)],
    "diagnosis_code": [diagnosis_code],
    "admission_type": [admission_type]
})

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict LOS"):
    pred = pipeline.predict(input_df)[0]

    # Absolute safety
    pred = np.nan_to_num(pred, nan=5)
    pred = int(round(pred))
    pred = max(1, min(9, pred))

    st.success(f"🛏️ Predicted Length of Stay: **{pred} days**")
