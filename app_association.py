def app():
    import streamlit as st
    st.title("Association Rule Mining")
import streamlit as st
import pickle
from collections import defaultdict

# =============================
# Load Association Rules
# =============================
with open(r"D:\Final_Project\ASSOCIATIVE LEARNING\association_rules.pkl", "rb") as f:
    rules = pickle.load(f)

st.set_page_config(page_title="Association Risk Analysis", layout="wide")
st.title("Real-World Association Risk Explorer")

# =============================
# UI: Risk Factor Selection
# =============================
st.sidebar.header("Select Patient Risk Factors")

bmi = st.sidebar.checkbox("BMI > 30 (Obesity)")
bp = st.sidebar.checkbox("High Blood Pressure")
diabetes = st.sidebar.checkbox("Diabetes")
smoking = st.sidebar.checkbox("Smoking")
chol = st.sidebar.checkbox("High Cholesterol")

age = st.sidebar.selectbox(
    "Age Group",
    ["< 30", "30–45", "45–60", "> 60"]
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# =============================
# Map UI → Rule Features
# =============================
selected_factors = set()

if bmi:
    selected_factors.add("obesity")
if bp:
    selected_factors.add("hypertension")
if diabetes:
    selected_factors.add("diabetes")
if smoking:
    selected_factors.add("smoking")
if chol:
    selected_factors.add("high_cholesterol")

age_map = {
    "< 30": "age_group_Young",
    "30–45": "age_group_Mid",
    "45–60": "age_group_Senior",
    "> 60": "age_group_Elderly"
}
selected_factors.add(age_map[age])
selected_factors.add(f"gender_{gender}")

# =============================
# Human-Readable Labels
# =============================
RISK_LABELS = {
    "obesity": "BMI > 30",
    "hypertension": "High Blood Pressure",
    "diabetes": "Diabetes",
    "smoking": "Smoking",
    "high_cholesterol": "High Cholesterol",
    "age_group_Young": "Age < 30",
    "age_group_Mid": "Age 30–45",
    "age_group_Senior": "Age 45–60",
    "age_group_Elderly": "Age > 60",
    "gender_Male": "Male",
    "gender_Female": "Female"
}

# =============================
# Disease Risk Logic
# =============================
def calculate_disease_risk(factors, confidence, disease):
    base = confidence * 100

    lung_weights = {
        "smoking": 0.30,
        "age_group_Senior": 0.20,
        "age_group_Elderly": 0.30,
        "obesity": 0.15
    }

    heart_weights = {
        "hypertension": 0.30,
        "diabetes": 0.25,
        "obesity": 0.20,
        "smoking": 0.15,
        "age_group_Senior": 0.25,
        "age_group_Elderly": 0.35,
        "gender_Male": 0.10
    }

    weights = lung_weights if disease == "lung" else heart_weights
    modifier = sum(weights.get(f, 0) for f in factors)

    return min(int(base + modifier * 100), 95)

# =============================
# Deduplicate & Aggregate Rules
# =============================
grouped_results = defaultdict(lambda: {
    "diabetes": 0,
    "lung": 0,
    "heart": 0
})

for _, rule in rules.iterrows():
    antecedents = frozenset(rule["antecedents"])

    if antecedents.issubset(selected_factors):
        conf = rule["confidence"]

        if "diabetes" in rule["consequents"]:
            grouped_results[antecedents]["diabetes"] = max(
                grouped_results[antecedents]["diabetes"],
                int(conf * 100)
            )

        if "smoking" in antecedents:
            grouped_results[antecedents]["lung"] = max(
                grouped_results[antecedents]["lung"],
                calculate_disease_risk(antecedents, conf, "lung")
            )

        if any(f in antecedents for f in ["hypertension", "diabetes", "obesity"]):
            grouped_results[antecedents]["heart"] = max(
                grouped_results[antecedents]["heart"],
                calculate_disease_risk(antecedents, conf, "heart")
            )
st.subheader("Association-Based Risk Results")

if not grouped_results:
    st.info("Select risk factors to see association-based risks.")
else:
    for antecedents, risks in grouped_results.items():
        readable = [RISK_LABELS.get(f, f) for f in antecedents]

        st.markdown("## " + " + ".join(readable))

        if risks["diabetes"] > 0:
            st.markdown(f"➡️ **Diabetes Risk {risks['diabetes']}%**")

        if risks["lung"] > 0:
            st.markdown(f"➡️ **Lung Disease Risk {risks['lung']}%**")

        if risks["heart"] > 0:
            st.markdown(f"➡️ **Heart Disease Risk {risks['heart']}%**")

        st.divider()            

