def app():
    import streamlit as st
    # --- EXISTING CODE BELOW (UNCHANGED) ---
    st.title("Risk Level Prediction")
import streamlit as st
import pandas as pd
import pickle

# ----------------------------------
# Load model
# ----------------------------------
with open(r"D:\Final_Project\RISK_LEVEL_CLASSIFICATION\risk_level_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
inverse_risk_map = data["inverse_risk_map"]
features = data["features"]

st.set_page_config(page_title="Patient Risk Level Predictor", layout="centered")
st.title("🩺 Patient Risk Level Prediction")

# ----------------------------------
# User Inputs
# ----------------------------------
age = st.number_input("Age", 1, 120, 40)

diet = st.selectbox("Diet", ["Poor", "Average", "Good"])
exercise_days = st.slider("Exercise Days per Week", 0, 7, 3)
sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
family_history = st.selectbox("Family History", ["No", "Yes"])

# ----------------------------------
# Encode categorical inputs
# ⚠ MUST match notebook encoding
# ----------------------------------
diet_map = {"Poor": 0, "Average": 1, "Good": 2}
stress_map = {"Low": 0, "Medium": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}

input_dict = {
    "age": age,
    "diet": diet_map[diet],
    "exercise_days": exercise_days,
    "sleep_hours": sleep_hours,
    "stress": stress_map[stress],
    "bmi": bmi,
    "smoking": yes_no_map[smoking],
    "alcohol": yes_no_map[alcohol],
    "family_history": yes_no_map[family_history]
}

# ----------------------------------
# Create DataFrame in SAME order
# ----------------------------------
input_df = pd.DataFrame([[input_dict[col] for col in features]],
                        columns=features)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Risk Level"):
    pred_encoded = model.predict(input_df)[0]
    pred_label = inverse_risk_map[pred_encoded]

    st.subheader("🧠 Prediction Result")

    if pred_label == "Low":
        st.success(f"Risk Level: **{pred_label}**")
    elif pred_label == "Medium":
        st.warning(f"Risk Level: **{pred_label}**")
    else:
        st.error(f"Risk Level: **{pred_label}**")
proba = model.predict_proba(input_df)[0]

st.write("### Prediction Confidence")
for i, p in enumerate(proba):
    st.write(f"{inverse_risk_map[i]}: {p:.2%}")