import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HealthAI Unified Dashboard",
    layout="wide"
)

st.title("🏥 HealthAI Unified Dashboard")
st.caption("Risk • LOS • Clustering • Association • ICU LSTM • Pneumonia • Sentiment")

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Risk Prediction",
        "Length of Stay",
        "Patient Clustering",
        "Association Rules",
        "ICU LSTM Prediction",
        "Pneumonia Detection",
        "Sentiment Analysis"
    ]
)

# =====================================================
# 1️⃣ RISK PREDICTION
# =====================================================
if menu == "Risk Prediction":
    st.subheader("🩺 Health Risk Level Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 45)
        diet = st.selectbox("Diet", ["Poor", "Average", "Good"])
        exercise_days = st.slider("Exercise Days / Week", 0, 7, 3)
        sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
        stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

    with col2:
        bmi = st.number_input("BMI", 15.0, 40.0, 24.5)
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol", ["Yes", "No"])
        family_history = st.selectbox("Family History", ["Yes", "No"])

    if st.button("Predict Risk"):
        payload = {
            "age": age,
            "diet": diet,
            "exercise_days": exercise_days,
            "sleep_hours": sleep_hours,
            "stress": stress,
            "bmi": bmi,
            "smoking": smoking,
            "alcohol": alcohol,
            "family_history": family_history
        }

        res = requests.post(f"{API_URL}/risk/predict", json=payload)
        if res.status_code == 200:
            st.success(f"Risk Level: **{res.json()['risk_level']}**")
        else:
            st.error(res.text)

# =====================================================
# 2️⃣ LOS
# =====================================================
elif menu == "Length of Stay":
    st.subheader("🏨 Length of Stay Prediction")

    age = st.number_input("Age", 18, 100, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    severity = st.slider("Severity", 1, 5, 3)
    comorbidities = st.slider("Comorbidities", 0, 5)
    procedure_code = st.number_input("Procedure Code",min_value=1,max_value=5,value=5)
    diagnosis_code = st.selectbox("Diagnosis Code",options=["D1", "D2", "D3", "D4", "D5"])
    admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])

    if st.button("Predict LOS"):
        payload = {
            "age": age,
            "gender": gender,
            "severity": severity,
            "comorbidities": comorbidities,
            "procedure_code": procedure_code,
            "diagnosis_code": diagnosis_code,
            "admission_type": admission_type
        }

        res = requests.post(f"{API_URL}/los/predict", json=payload)
        st.success(f"Predicted LOS: **{res.json()['length_of_stay_days']} days**")

# =====================================================
# 3️⃣ CLUSTERING
# =====================================================
elif menu == "Patient Clustering":
    st.subheader("👥 Patient Clustering")

    age = st.number_input("Age", 18, 100, 55)
    bmi = st.number_input("BMI", 15.0, 45.0, 26.0)
    chronic_conditions = st.slider("Chronic Conditions", 0, 5, 2)
    visit_frequency = st.slider("Visits / Year", 0, 20, 6)
    avg_stay_days = st.slider("Avg Stay Days", 1, 20, 5)
    icu_admissions = st.slider("ICU Admissions", 0, 5, 1)
    emergency_visits = st.slider("Emergency Visits", 0, 10, 2)

    if st.button("Predict Cluster"):
        payload = {
            "age": age,
            "bmi": bmi,
            "chronic_conditions": chronic_conditions,
            "visit_frequency": visit_frequency,
            "avg_stay_days": avg_stay_days,
            "icu_admissions": icu_admissions,
            "emergency_visits": emergency_visits
        }

        res = requests.post(f"{API_URL}/cluster/predict", json=payload)
        st.info(res.json())

# =====================================================
# 4️⃣ ASSOCIATION RULES
# =====================================================
elif menu == "Association Rules":
    st.subheader("🔗 Disease Association Rules")

    cols = st.columns(5)
    hypertension = cols[0].checkbox("Hypertension")
    diabetes = cols[1].checkbox("Diabetes")
    obesity = cols[2].checkbox("Obesity")
    smoking = cols[3].checkbox("Smoking")
    high_cholesterol = cols[4].checkbox("High Cholesterol")

    if st.button("Find Associations"):
        payload = {
            "hypertension": int(hypertension),
            "diabetes": int(diabetes),
            "obesity": int(obesity),
            "smoking": int(smoking),
            "high_cholesterol": int(high_cholesterol)
        }

        res = requests.post(f"{API_URL}/associate/predict", json=payload)
        rules = res.json()["matched_rules"]

        if rules:
            st.dataframe(pd.DataFrame(rules))
        else:
            st.warning("No matching rules found")

# =====================================================
# 5️⃣ LSTM ICU
# =====================================================
elif menu == "ICU LSTM Prediction":
    st.subheader("❤️ ICU Heart Rate Prediction (24×6)")

    st.info("Enter last 24 hours vitals (heart_rate, sbp, dbp, spo2, rr, temp)")

    data = np.random.normal([80, 120, 80, 97, 18, 36.8], [5,5,5,1,2,0.3], (24,6))
    vitals = st.data_editor(pd.DataFrame(data), height=400)

    if st.button("Predict ICU Risk"):
        payload = {"vitals": vitals.values.tolist()}
        res = requests.post(f"{API_URL}/lstm/predict", json=payload)
        st.json(res.json())

# =====================================================
# 6️⃣ PNEUMONIA
# =====================================================
elif menu == "Pneumonia Detection":
    st.subheader("🫁 Pneumonia Detection from Chest X-ray")

    file = st.file_uploader("Upload X-ray Image", type=["jpg","png"])

    if file and st.button("Predict Pneumonia"):
        files = {"file": file.getvalue()}
        res = requests.post(
            f"{API_URL}/pneumonia/predict",
            files={"file": (file.name, file, file.type)}
        )
        st.image(Image.open(file), width=300)
        st.success(res.json())

# =====================================================
# 7️⃣ SENTIMENT
# =====================================================
elif menu == "Sentiment Analysis":
    st.subheader("💬 Patient Feedback Sentiment")

    text = st.text_area("Enter patient feedback")

    if st.button("Analyze Sentiment"):
        res = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )
        st.success(res.json())
