def app():
    import streamlit as st
    # --- EXISTING CODE BELOW (UNCHANGED) ---
    st.title("LSTM Time-Series Prediction")
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="ICU Deterioration Prediction",
    layout="wide"
)

st.title("🏥 ICU Deterioration Prediction (LSTM)")
st.write(
    """
    This application predicts **next-hour heart rate** using an LSTM model
    trained on 24-hour ICU vital time-series and maps it to a **clinical ICU risk level**.
    """
)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
model = load_model(r"D:\Final_Project\LSTM\icu_lstm_hr_model.h5")
scaler = joblib.load(r"D:\Final_Project\LSTM\icu_scaler.joblib")

# --------------------------------------------------
# Clinical Risk Logic (YOUR FUNCTION)
# --------------------------------------------------
def icu_risk_level(pred_hr, spo2):
    if pred_hr > 120 or spo2 < 90:
        return "🔴 Critical"
    elif pred_hr > 100 or spo2 < 94:
        return "🟠 Warning"
    else:
        return "🟢 Stable"

# --------------------------------------------------
# Feature Order (MUST match training)
# --------------------------------------------------
FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature"
]

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("🕒 Enter Last 24 Hours ICU Vitals")

st.caption("Each row represents one hour (most recent hour = Hour 24)")

input_data = []

for hour in range(24):
    st.markdown(f"**Hour {hour + 1}**")
    cols = st.columns(6)

    hr = cols[0].number_input(
        "Heart Rate (bpm)", min_value=40, max_value=180, value=80, key=f"hr_{hour}"
    )
    sbp = cols[1].number_input(
        "Systolic BP", min_value=80, max_value=220, value=120, key=f"sbp_{hour}"
    )
    dbp = cols[2].number_input(
        "Diastolic BP", min_value=40, max_value=130, value=80, key=f"dbp_{hour}"
    )
    spo2 = cols[3].number_input(
        "SpO₂ (%)", min_value=85, max_value=100, value=97, key=f"spo2_{hour}"
    )
    rr = cols[4].number_input(
        "Respiratory Rate", min_value=10, max_value=40, value=18, key=f"rr_{hour}"
    )
    temp = cols[5].number_input(
        "Temperature (°C)", min_value=35.0, max_value=40.0, value=36.8, key=f"temp_{hour}"
    )

    input_data.append([hr, sbp, dbp, spo2, rr, temp])

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Next Hour & ICU Risk"):

    # Convert to numpy
    input_array = np.array(input_data)

    # Scale input
    scaled_input = scaler.transform(input_array)

    # Reshape for LSTM → (1, 24, 6)
    X_input = scaled_input.reshape(1, 24, len(FEATURES))

    # Predict scaled HR
    scaled_pred_hr = model.predict(X_input, verbose=0)[0][0]

    # 🔑 Inverse transform ONLY heart rate
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = scaled_pred_hr   # heart_rate index = 0
    pred_hr = scaler.inverse_transform(dummy)[0][0]

    # Last hour SpO₂ (for risk logic)
    last_spo2 = input_data[-1][3]

    # Risk assessment
    risk = icu_risk_level(pred_hr, last_spo2)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Predicted Next-Hour HR", f"{pred_hr:.2f} bpm")
    col2.metric("🫁 Last SpO₂", f"{last_spo2}%")
    col3.metric("⚠️ ICU Risk Level", risk)

    if "Critical" in risk:
        st.error("🚨 High risk of ICU deterioration. Immediate clinical attention recommended.")
    elif "Warning" in risk:
        st.warning("⚠️ Patient requires close monitoring.")
    else:
        st.success("✅ Patient appears clinically stable.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Model: LSTM (24-hour multivariate ICU time-series) | "
    "Decision Layer: Rule-based clinical logic"
)
