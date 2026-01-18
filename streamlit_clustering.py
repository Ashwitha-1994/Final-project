def app():
    import streamlit as st
    # --- EXISTING CODE BELOW (UNCHANGED) ---
    st.title("Patient Clustering")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* White card for readability */
        section.main > div {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 12px;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: rgba(240, 245, 250, 0.95);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def apply_dark_ui():
    st.markdown(
        """
        <style>
        /* MAIN APP BACKGROUND */
        .stApp {
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* ================================
           SIDEBAR (INPUTS) – BLACK THEME
        ================================= */
        section[data-testid="stSidebar"] {
            background-color: #0b0f14 !important;
        }

        section[data-testid="stSidebar"] * {
            color: #e6e6e6 !important;
        }

        /* Sliders */
        .stSlider > div {
            color: white !important;
        }

        /* Selectbox & inputs */
        div[data-baseweb="select"],
        input {
            background-color: #1c1f26 !important;
            color: white !important;
        }

        /* ================================
           MAIN CONTENT – DARK GLASS CARD
        ================================= */
        section.main > div {
            background: rgba(10, 15, 25, 0.88);
            padding: 2.5rem;
            border-radius: 16px;
            color: #f2f2f2;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }

        /* SUCCESS (Prediction result) */
        .stSuccess {
            background-color: rgba(0, 128, 96, 0.85) !important;
            color: white !important;
            border-radius: 10px;
        }

        /* INFO (Interpretation) */
        .stInfo {
            background-color: rgba(30, 58, 138, 0.85) !important;
            color: white !important;
            border-radius: 10px;
        }

        /* Buttons */
        button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def enhance_output_visibility():
    st.markdown(
        """
        <style>
        /* ================================
           PREDICTION RESULT – STRONG DARK
        ================================= */
        .stSuccess {
            background: linear-gradient(
                135deg,
                rgba(0, 90, 60, 0.95),
                rgba(0, 120, 80, 0.95)
            ) !important;
            color: #eafff7 !important;
            border-radius: 14px;
            padding: 1.2rem;
            font-size: 1.15rem;
            font-weight: 600;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        }

        /* ================================
           INTERPRETATION BOX – DARK BLUE
        ================================= */
        .stInfo {
            background: linear-gradient(
                135deg,
                rgba(10, 30, 90, 0.95),
                rgba(20, 50, 120, 0.95)
            ) !important;
            color: #f0f6ff !important;
            border-radius: 14px;
            padding: 1.2rem;
            font-size: 1.05rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        }

        /* Make output section headings pop */
        h2, h3 {
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Patient Risk Clustering", layout="centered")
st.title("🧠 Patient Risk Clustering System")
add_bg_from_local(r"D:\Final_Project\Clustering_codes\hospital_bg.jpg")
apply_dark_ui()
enhance_output_visibility()

# -------------------------------
# LOAD PIPELINE
# -------------------------------
@st.cache_resource
def load_pipeline():
    with open(r"D:\Final_Project\Clustering_codes\patient_clustering_pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

features = pipeline["features"]
scaler = pipeline["scaler"]
kmeans = pipeline["kmeans"]
pca = pipeline["pca"]
cluster_labels = pipeline["cluster_labels"]

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🧾 Patient Details")

age = st.sidebar.slider("Age", 18, 90, 45)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
chronic = st.sidebar.slider("Chronic Conditions", 0, 7, 1)
visits = st.sidebar.slider("Hospital Visits / Year", 0, 20, 2)
stay = st.sidebar.slider("Avg Stay (days)", 1, 30, 3)
icu = st.sidebar.selectbox("ICU Admission", [0, 1])
emergency = st.sidebar.slider("Emergency Visits", 0, 15, 1)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.sidebar.button("🔍 Predict Patient Group"):

    input_df = pd.DataFrame([[
        age, bmi, chronic, visits, stay, icu, emergency
    ]], columns=features)

    input_scaled = scaler.transform(input_df)
    cluster_id = kmeans.predict(input_scaled)[0]
    cluster_name = cluster_labels[cluster_id]

    # CLINICAL RULE OVERRIDES (CRITICAL)
    # =====================================================
    if age < 40 and chronic <= 1 and visits <= 2 and icu == 0:
        cluster_name = "Young Healthy"

    elif icu == 1 or emergency >= 6:
        cluster_name = "High-Acuity"

    elif age >= 65 or chronic >= 3:
        cluster_name = "Elderly Chronic"

    elif 40 <= age < 65:
        cluster_name = "Middle-Aged Preventive"

    st.subheader("🧩 Prediction Result")
    st.success(f"**Patient belongs to: {cluster_name}**")

    # -------------------------------
    # CLINICAL INTERPRETATION
    # -------------------------------
    interpretations = {
        "Young Healthy": "Low-risk patients with minimal medical intervention.",
        "Middle-Aged Preventive": "Moderate risk patients needing regular monitoring.",
        "Elderly Chronic": "High-risk elderly patients with chronic illnesses.",
        "High-Acuity": "Critical patients requiring ICU and complex care."
    }

    st.info(interpretations[cluster_name])


