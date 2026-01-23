from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List
import pandas as pd
import numpy as np
import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app_final = FastAPI(
    title="HealthAI Unified API",
    description="Risk, LOS, Clustering, Association & Time-Series Prediction",
    version="1.0.0"
)

# =========================================================
# LOAD MODELS
# =========================================================
import joblib
import pickle

# Risk model (pickle)
with open(r"D:\Final_Project\RISK_LEVEL_CLASSIFICATION\risk_level_model.pkl", "rb") as f:
    risk_data = pickle.load(f)

risk_model = risk_data["model"]
risk_features = risk_data["features"]
inverse_risk_map = risk_data["inverse_risk_map"]

# LOS model (joblib) ✅ FIXED
los_pipeline = joblib.load(r"D:\Final_Project\LOS\los_final.pkl")

# Clustering model (pickle)
# ---------------- CLUSTERING ----------------
with open(r"D:\Final_Project\Clustering_codes\patient_clustering_pipeline.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

# 🔥 THESE LINES WERE MISSING
cluster_features = cluster_pipeline["features"]
cluster_scaler = cluster_pipeline["scaler"]
cluster_kmeans = cluster_pipeline["kmeans"]
cluster_labels = cluster_pipeline["cluster_labels"]



# Association rules
with open(r"D:\Final_Project\ASSOCIATIVE LEARNING\association_rules.pkl", "rb") as f:
    association_rules = pickle.load(f)
    print("\n=== Association rules loaded ===")
for rule, conf in association_rules.items():
    print(rule, conf)
print("=== End of rules ===\n")

# Load Model & Scaler (SAME AS STREAMLIT)
# --------------------------------------------------
lstm_model = load_model(r"D:\Final_Project\LSTM\icu_lstm_hr_model.h5")
scaler = joblib.load(r"D:\Final_Project\LSTM\icu_scaler.joblib")
TIMESTEPS = 24
FEATURES = 6
FEATURE_NAMES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature"
]
# --------------------------------------------------
# Risk Logic (SAME AS STREAMLIT)
# --------------------------------------------------
def icu_risk_level(pred_hr, spo2):
    if pred_hr > 120 or spo2 < 90:
        return "🔴 Critical"
    elif pred_hr > 100 or spo2 < 94:
        return "🟠 Warning"
    else:
        return "🟢 Stable"
#CNN  MODEL PATH
CNN_MODEL_PATH = r"D:\Final_Project\CNN\mobilenet_pneumonia.h5"    
# Load Model (ONCE)
# --------------------------------------------------
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")
# Image Preprocessing (SAME AS STREAMLIT)
# --------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# Model path
# -------------------------------
SENTI_MODEL_PATH = r"D:\Final_Project\SENTIMENT_ANALYSIS\sentiment_transformer"
tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_PATH)
senti_model = AutoModelForSequenceClassification.from_pretrained(
    SENTI_MODEL_PATH,
    torch_dtype=torch.float32
)

senti_model.to("cpu")
senti_model.eval()
# Label mapping
# -------------------------------
label_map = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}


# =========================================================
# SCHEMAS
# =========================================================

class RiskInput(BaseModel):
    age: int
    diet: Literal["Poor","Average","Good"]
    exercise_days: int
    sleep_hours: float
    stress: Literal["Low","Medium","High"]
    bmi: float
    smoking: Literal["Yes","No"]
    alcohol: Literal["Yes","No"]
    family_history: Literal["Yes","No"]

class LOSInput(BaseModel):
    age: int
    gender: str
    severity: int
    comorbidities: int
    procedure_code: int
    diagnosis_code: str
    admission_type: str

class ClusterInput(BaseModel):
    age: int
    bmi: float
    chronic_conditions: int
    visit_frequency: int   # ✅ FIXED
    avg_stay_days: int
    icu_admissions: int
    emergency_visits: int
EXPECTED_FEATURES = 6
EXPECTED_TIMESTEPS = 24 # change if your model differs    


class AssocInput(BaseModel):
    hypertension: int
    diabetes: int
    obesity: int
    smoking: int
    high_cholesterol: int

from pydantic import BaseModel
from typing import List

class LSTMInput(BaseModel):
    vitals: list  # shape: (24, 6)

class FeedbackRequest(BaseModel):
    text: str

# -------------------------------
# Response schema
# -------------------------------
class SentimentResponse(BaseModel):
    sentiment: str
    label_id: int    

# =========================================================
# RISK LEVEL
# =========================================================
@app_final.post("/risk/predict")
def predict_risk(data: RiskInput):
    try:
        # -----------------------------
        # Encode categorical variables
        # -----------------------------
        diet_map = {"Poor": 0, "Average": 1, "Good": 2}
        stress_map = {"Low": 0, "Medium": 1, "High": 2}
        yes_no_map = {"No": 0, "Yes": 1}
        input_dict = {
            "age": data.age,
            "diet": diet_map[data.diet],
            "exercise_days": data.exercise_days,
            "sleep_hours": data.sleep_hours,
            "stress": stress_map[data.stress],
            "bmi": data.bmi,
            "smoking": yes_no_map[data.smoking],
            "alcohol": yes_no_map[data.alcohol],
            "family_history": yes_no_map[data.family_history]
        }
        input_df = pd.DataFrame(
        [[input_dict[col] for col in risk_features]],
        columns=risk_features
        )

        # -----------------------------
        # Predict
        # -----------------------------
        pred_encoded = risk_model.predict(input_df)[0]
        pred_label = inverse_risk_map[pred_encoded]

        return {
            "risk_level": pred_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# LOS
# =========================================================
@app_final.post("/los/predict")
def predict_los(data: LOSInput):
    df = pd.DataFrame([data.dict()])
    pred = los_pipeline.predict(df)[0]
    pred = int(max(1, min(15, round(pred))))
    return {"length_of_stay_days": pred}

# =========================================================
# CLUSTERING
# =========================================================
@app_final.post("/cluster/predict")
def predict_cluster(data: ClusterInput):
    try:
        # STRICT feature order (must match training)
        input_dict = {
            "age": data.age,
            "bmi": data.bmi,
            "chronic_conditions": data.chronic_conditions,
            "visit_frequency": data.visit_frequency,
            "avg_stay_days": data.avg_stay_days,
            "icu_admissions": data.icu_admissions,
            "emergency_visits": data.emergency_visits
        }

        # Create DataFrame EXACTLY as during fit
        input_df = pd.DataFrame(
            [[input_dict[col] for col in cluster_features]],
            columns=cluster_features
        )

        # Scale with CLUSTER scaler
        input_scaled = cluster_scaler.transform(input_df)

        # Predict cluster
        cluster_id = int(cluster_kmeans.predict(input_scaled)[0])

        cluster_name = (
            cluster_labels[cluster_id]
            if isinstance(cluster_labels, dict)
            else str(cluster_labels[cluster_id])
        )

        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =========================================================
# ASSOCIATION RULES
# =========================================================
@app_final.post("/associate/predict")
def predict_association(data: AssocInput):
    try:
        # Patient conditions
        patient = set(
            k for k, v in data.dict().items() if v == 1
        )

        matches = []

        # Iterate over RULE ROWS (not columns)
        for _, row in association_rules.iterrows():
            antecedents = set(row["antecedents"])
            consequents = set(row["consequents"])

            # Rule fires if antecedents ⊆ patient
            if antecedents.issubset(patient):
                matches.append({
                    "if_conditions": list(antecedents),
                    "then_conditions": list(consequents),
                    "support": round(row["support"], 3),
                    "confidence": round(row["confidence"], 3),
                    "lift": round(row["lift"], 3)
                })

        # Sort by confidence (best rules first)
        matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)

        return {
            "matched_rules": matches[:5]  # top-5 rules
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# TIME SERIES (LSTM)
# =========================================================
import numpy as np
import torch

@app_final.post("/lstm/predict")
def predict_lstm(data: LSTMInput):

    vitals = np.array(data.vitals, dtype=np.float32)

    # Validate shape
    if vitals.shape != (TIMESTEPS, FEATURES):
        raise HTTPException(
            status_code=400,
            detail="Expected input shape: (24, 6) → 24 hours × 6 vitals"
        )

    # --------------------------------------------------
    # SAME PIPELINE AS STREAMLIT
    # --------------------------------------------------

    # Scale input
    scaled_input = scaler.transform(vitals)

    # Reshape for LSTM
    X_input = scaled_input.reshape(1, 24, FEATURES)

    # Predict scaled HR
    scaled_pred_hr = lstm_model.predict(X_input, verbose=0)[0][0]

    # 🔑 Inverse transform ONLY heart rate (index 0)
    dummy = np.zeros((1, FEATURES))
    dummy[0, 0] = scaled_pred_hr   # heart_rate index = 0
    pred_hr = scaler.inverse_transform(dummy)[0][0]

    # Last hour SpO₂
    last_spo2 = vitals[-1][3]

    # Risk assessment
    risk = icu_risk_level(pred_hr, last_spo2)

    # --------------------------------------------------
    # Response
    # --------------------------------------------------
    return {
        "predicted_heart_rate": round(float(pred_hr), 2),
        "unit": "bpm",
        "last_spo2": float(last_spo2),
        "icu_risk_level": risk,
        "timesteps_used": 24,
        "model": "LSTM ICU Time Series",
        "features": FEATURE_NAMES
    }
@app_final.post("/pneumonia/predict")
async def predict_pneumonia(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG or PNG images are supported"
        )

    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess
        input_tensor = preprocess_image(image)

        # Predict
        pred = cnn_model.predict(input_tensor, verbose=0)
        prob = float(pred[0][0])

        # Decision
        if prob >= 0.5:
            label = "PNEUMONIA"
            confidence = round(prob, 3)
        else:
            label = "NORMAL"
            confidence = round(1 - prob, 3)

        return {
            "prediction": label,
            "confidence": confidence,
            "model": "MobileNet CNN",
            "input_size": "224x224",
            "threshold": 0.5
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_final.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: FeedbackRequest):

    if not request.text.strip():
        return {"sentiment": "Empty input", "label_id": -1}

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = senti_model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=1).item()

    return {
        "sentiment": label_map[pred],
        "label_id": pred
    }



