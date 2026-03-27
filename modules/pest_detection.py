import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import requests

# ================== CONFIG ==================

MODEL_URL = "https://huggingface.co/bhaskar06/plant-model-1/resolve/main/plant_disease_model%20(1).h5"
MODEL_PATH = "/tmp/plant_disease_model.keras"
IMG_SIZE = (224, 224)

# ============================================

@st.cache_resource
def load_model():
    """Download model from HuggingFace (first time only) and load it."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model (first time only)..."):
            resp = requests.get(MODEL_URL, stream=True, timeout=120)

            st.write("HTTP status:", resp.status_code)
            st.write("Final URL:", resp.url)

            if resp.status_code != 200:
                st.error(f"❌ Failed to download model. Status: {resp.status_code}")
                st.stop()

            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_class_names():
    """Load class index mapping from class_indices.json (in project root)."""
    json_path = os.path.join(os.path.dirname(__file__), '..', 'class_indices.json')
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


# ================== ADVICE MAPS ==================

# Simple keyword-based advice (used by get_advice())
DISEASE_ADVICE = {
    "healthy":                  "✅ Your plant is healthy! No action needed.",
    "Apple_scab":               "🍎 Apply fungicides like Captan or Mancozeb. Remove infected leaves.",
    "Black_rot":                "🍇 Prune infected parts and use copper-based fungicide.",
    "Early_blight":             "🍅 Use Chlorothalonil fungicide. Avoid overhead watering.",
    "Late_blight":              "🥔 Apply Metalaxyl fungicide immediately, remove infected plants.",
    "Powdery_mildew":           "🍃 Use sulfur-based fungicide and improve air circulation.",
    "Bacterial_spot":           "🫑 Use copper bactericide spray. Avoid working with wet plants.",
    "Leaf_Mold":                "🍅 Improve ventilation and apply Mancozeb.",
    "Septoria_leaf_spot":       "🍅 Remove lower leaves and apply Chlorothalonil.",
    "Yellow_Leaf_Curl_Virus":   "🍅 Control whiteflies and remove infected plants.",
}

# Exact-label advice map (used by get_pest_advice())
PEST_ADVICE_MAP = {
    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple___Apple_scab":                       "🍎 Use sulfur spray. Prune infected branches. Apply Captan before rain.",
    "Apple___Black_rot":                        "🍎 Remove mummified fruits. Apply copper-based fungicide every 10 days.",
    "Apple___Cedar_apple_rust":                 "🍎 Apply Myclobutanil fungicide at bud break. Remove nearby cedar trees if possible.",
    "Apple___healthy":                          "✅ Apple plant is healthy. Maintain regular watering and pruning.",

    # ── Blueberry ──────────────────────────────────────────────────────────
    "Blueberry___healthy":                      "✅ Blueberry plant is healthy. Ensure acidic soil (pH 4.5–5.5).",

    # ── Cherry ─────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": "🍒 Apply sulfur-based fungicide. Improve air circulation between plants.",
    "Cherry_(including_sour)___healthy":        "✅ Cherry plant is healthy. Continue good orchard hygiene.",

    # ── Corn (Maize) ───────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "🌽 Apply Azoxystrobin fungicide. Rotate crops annually.",
    "Corn_(maize)___Common_rust_":              "🌽 Apply Propiconazole fungicide. Plant resistant varieties next season.",
    "Corn_(maize)___Northern_Leaf_Blight":      "🌽 Use Mancozeb fungicide. Ensure proper field drainage.",
    "Corn_(maize)___healthy":                   "✅ Corn plant is healthy. Maintain adequate nitrogen fertilization.",

    # ── Grape ──────────────────────────────────────────────────────────────
    "Grape___Black_rot":                        "🍇 Remove infected berries immediately. Apply Mancozeb or Captan fungicide.",
    "Grape___Esca_(Black_Measles)":             "🍇 Prune infected wood. No chemical cure — prevent via wound protection paste.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "🍇 Apply copper-based fungicide. Remove fallen infected leaves.",
    "Grape___healthy":                          "✅ Grape vine is healthy. Maintain proper canopy management.",

    # ── Orange ─────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": "🍊 No cure exists. Remove infected trees. Control Asian citrus psyllid with insecticide.",

    # ── Peach ──────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot":                   "🍑 Apply copper bactericide spray early season. Avoid overhead irrigation.",
    "Peach___healthy":                          "✅ Peach plant is healthy. Thin fruits to improve air circulation.",

    # ── Pepper ─────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot":            "🫑 Use copper-based bactericide. Rotate crops. Avoid working in wet conditions.",
    "Pepper,_bell___healthy":                   "✅ Bell pepper is healthy. Ensure consistent watering to prevent blossom drop.",

    # ── Potato ─────────────────────────────────────────────────────────────
    "Potato___Early_blight":                    "🥔 Apply Chlorothalonil or Mancozeb fungicide. Remove lower infected leaves.",
    "Potato___Late_blight":                     "🥔 Apply Metalaxyl fungicide immediately. Destroy infected plants. Do not compost.",
    "Potato___healthy":                         "✅ Potato plant is healthy. Hill soil around plants to prevent greening.",

    # ── Raspberry ──────────────────────────────────────────────────────────
    "Raspberry___healthy":                      "✅ Raspberry plant is healthy. Prune old canes after harvest.",

    # ── Soybean ────────────────────────────────────────────────────────────
    "Soybean___healthy":                        "✅ Soybean plant is healthy. Monitor for aphids and spider mites.",

    # ── Squash ─────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew":                  "🎃 Apply potassium bicarbonate or sulfur spray. Increase plant spacing.",

    # ── Strawberry ─────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch":                 "🍓 Remove infected leaves. Apply Captan fungicide. Avoid overhead watering.",
    "Strawberry___healthy":                     "✅ Strawberry plant is healthy. Renew beds every 3–4 years.",

    # ── Tomato ─────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot":                  "🍅 Apply copper-based bactericide. Use disease-free seeds. Avoid leaf wetness.",
    "Tomato___Early_blight":                    "🍅 Apply Chlorothalonil. Remove lower infected leaves. Mulch around base.",
    "Tomato___Late_blight":                     "🍅 Apply copper-based fungicide. Remove infected leaves. Avoid overhead watering.",
    "Tomato___Leaf_Mold":                       "🍅 Improve greenhouse ventilation. Apply Mancozeb or Chlorothalonil.",
    "Tomato___Septoria_leaf_spot":              "🍅 Remove lower leaves. Apply Chlorothalonil every 7–10 days.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "🍅 Apply Neem oil or Abamectin miticide. Increase humidity around plants.",
    "Tomato___Target_Spot":                     "🍅 Apply Azoxystrobin fungicide. Ensure good air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":   "🍅 Control whitefly populations with insecticide. Remove and destroy infected plants.",
    "Tomato___Tomato_mosaic_virus":             "🍅 No chemical cure. Remove infected plants. Disinfect tools with bleach solution.",
    "Tomato___healthy":                         "✅ Tomato plant is healthy. Stake plants and prune suckers regularly.",
}


def get_advice(class_name: str) -> str:
    """Keyword-based advice fallback."""
    for key, advice in DISEASE_ADVICE.items():
        if key.lower() in class_name.lower():
            return advice
    if "healthy" in class_name.lower():
        return DISEASE_ADVICE["healthy"]
    return "⚠️ Please consult a local agricultural expert for precise treatment."


def get_pest_advice(disease_label: str) -> str:
    """
    Returns detailed treatment advice for a given disease label.
    Tries exact match first, then partial keyword match, then fallback.
    """
    # 1. Exact match
    if disease_label in PEST_ADVICE_MAP:
        return PEST_ADVICE_MAP[disease_label]

    # 2. Partial match (handles minor label variations)
    for key, advice in PEST_ADVICE_MAP.items():
        if key.lower() in disease_label.lower() or disease_label.lower() in key.lower():
            return advice

    # 3. Healthy catch-all
    if "healthy" in disease_label.lower():
        return "✅ Plant is healthy! Maintain good watering, fertilization, and monitoring practices."

    # 4. Final fallback
    return "⚠️ Consult an agronomist for this disease. Avoid applying chemicals without proper diagnosis."


# ================== IMAGE PREPROCESSING ==================

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for the model."""
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ================== MAIN PAGE ==================

def pest_detection_page():
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload a **leaf image** to detect diseases using AI (**98.71% accuracy model**).")

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
                probs = preds[0]
                idx = int(np.argmax(probs))
                confidence = float(np.max(probs) * 100.0)
                class_name = class_names[idx]

            # Split class name into plant & disease
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

            # ── Treatment Advice (uses get_pest_advice for exact match first) ──
            st.subheader("💊 Treatment Advice")
            advice = get_pest_advice(class_name)
            # fallback to keyword-based if pest advice returned generic message
            if "Consult an agronomist" in advice:
                advice = get_advice(class_name)
            st.info(advice)

            st.markdown("---")

            # ── Top 3 Predictions ──
            st.subheader("📊 Top 3 Predictions")
            top3_idx = np.argsort(probs)[::-1][:3]
            for rank, i in enumerate(top3_idx, start=1):
                cname = class_names[int(i)].replace("___", " → ").replace("_", " ")
                prob = float(probs[int(i)] * 100.0)
                st.progress(int(prob), text=f"{rank}. {cname}: {prob:.1f}%")

            st.markdown("---")

            # ── Pest Advice for Top 3 ──
            st.subheader("🌾 Detailed Advice for Top Predictions")
            for rank, i in enumerate(top3_idx, start=1):
                raw_label = class_names[int(i)]
                prob = float(probs[int(i)] * 100.0)
                display = raw_label.replace("___", " → ").replace("_", " ")
                with st.expander(f"{rank}. {display} ({prob:.1f}%)"):
                    st.write(get_pest_advice(raw_label))