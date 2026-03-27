import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import requests

# ---------- CONFIG ----------

# Put your real HF URL here:
MODEL_URL = "https://huggingface.co/bhaskar06/plant-model-1/tree/main/plant_disease_model,h5"
MODEL_PATH = "/tmp/plant_disease_model.h5"  # temp location on Streamlit Cloud

# ----------------------------

@st.cache_resource
def load_model():
    """Download model from HuggingFace (first time only) and load it."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model (first time only)..."):
            resp = requests.get(MODEL_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    """Load class index mapping from class_indices.json"""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'class_indices.json')
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    # Reverse mapping: {index: class_name}
    return {v: k for k, v in class_indices.items()}

# Simple advice mapping (you can expand later)
DISEASE_ADVICE = {
    "healthy": "✅ Your plant is healthy! No action needed.",
    "Apple_scab": "🍎 Apply fungicides like Captan or Mancozeb. Remove infected leaves.",
    "Black_rot": "🍇 Prune infected parts and use copper-based fungicide.",
    "Early_blight": "🍅 Use Chlorothalonil fungicide. Avoid overhead watering.",
    "Late_blight": "🥔 Apply Metalaxyl fungicide immediately, remove infected plants.",
    "Powdery_mildew": "🍃 Use sulfur-based fungicide and improve air circulation.",
    "Bacterial_spot": "🫑 Copper bactericide spray, avoid working with wet plants.",
    "Leaf_Mold": "🍅 Improve ventilation and apply Mancozeb.",
    "Septoria_leaf_spot": "🍅 Remove lower leaves, apply Chlorothalonil.",
    "Yellow_Leaf_Curl_Virus": "🍅 Control whiteflies, remove infected plants."
}

def get_advice(class_name: str) -> str:
    for key, advice in DISEASE_ADVICE.items():
        if key.lower() in class_name.lower():
            return advice
    if "healthy" in class_name.lower():
        return DISEASE_ADVICE["healthy"]
    return "⚠️ Please consult a local agricultural expert for precise treatment."

def preprocess_image(img: Image.Image) -> np.ndarray:
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def pest_detection_page():
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload a **leaf image** to detect diseases using AI (**98.71% accuracy model**).")

    # Load resources
    model = load_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "📷 Upload Leaf Image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_column_width=True)

        with col2:
            st.subheader("🤖 AI Diagnosis")

            x = preprocess_image(img)

            with st.spinner("Analyzing leaf..."):
                preds = model.predict(x)
                idx = np.argmax(preds[0])
                confidence = float(np.max(preds[0]) * 100.0)
                class_name = class_names[idx]

            # Split class into plant and disease
            parts = class_name.split("___")
            plant = parts[0].replace("_", " ")
            disease = (parts[1] if len(parts) > 1 else "Unknown").replace("_", " ")

            st.markdown(f"### 🌿 Plant: `{plant}`")
            st.markdown(f"### 🦠 Condition: `{disease}`")

            if confidence >= 90:
                st.success(f"✅ Confidence: **{confidence:.2f}%** (High)")
            elif confidence >= 70:
                st.warning(f"⚠️ Confidence: **{confidence:.2f}%** (Medium)")
            else:
                st.error(f"❌ Confidence: **{confidence:.2f}%** (Low — try a clearer image)")

            st.markdown("---")
            st.subheader("💊 Treatment Advice")
            st.info(get_advice(class_name))

            st.markdown("---")
            st.subheader("📊 Top 3 Predictions")
            top3_idx = np.argsort(preds[0])[::-1][:3]
            for rank, i in enumerate(top3_idx, start=1):
                cname = class_names[i].replace("___", " → ").replace("_", " ")
                prob = float(preds[0][i] * 100.0)
                st.progress(int(prob), text=f"{rank}. {cname}: {prob:.1f}%")