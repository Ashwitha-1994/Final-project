def app():
    import streamlit as st
    # --- EXISTING CODE BELOW (UNCHANGED) ---
    st.title("Sentiment Analysis")
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = r"D:\Final_Project\SENTIMENT_ANALYSIS\sentiment_transformer"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map=None   # 🚨 IMPORTANT
    )
    model.to("cpu")      # 🚨 FORCE real tensors
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

label_map = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}

st.title("🏥 Hospital Feedback Sentiment Analysis")

text = st.text_area("Enter patient feedback")

if st.button("Analyze"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # 🚨 FORCE inputs to CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(dim=1).cpu().numpy()[0]  # ✅ SAFE

        st.success(f"Sentiment: **{label_map[pred]}**")
