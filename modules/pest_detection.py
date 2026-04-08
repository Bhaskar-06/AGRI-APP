import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import requests

# ================== CONFIG ==================

MODEL_URL = "https://huggingface.co/bhaskar06/plant-model-1/resolve/main/plant_disease_model%20(1).h5"
MODEL_PATH = "/tmp/plant_disease_model.h5"
IMG_SIZE = (224, 224)

# ============================================

def download_model():
    """Download model from HuggingFace with validation."""
    if os.path.exists(MODEL_PATH):
        # Validate file size (should be > 50MB for a real model)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb < 10:
            st.warning(f"⚠️ Cached model seems too small ({size_mb:.1f} MB). Re-downloading...")
            os.remove(MODEL_PATH)
        else:
            return True  # Already downloaded and valid

    with st.spinner(f"⬇️ Downloading AI model (first time only)..."):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(MODEL_URL, stream=True, timeout=180, headers=headers)

            if resp.status_code != 200:
                st.error(f"❌ Download failed. HTTP Status: {resp.status_code}")
                st.error(f"URL tried: {resp.url}")
                return False

            total = int(resp.headers.get('content-length', 0))
            progress = st.progress(0, text="Downloading model...")

            downloaded = 0
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = min(int(downloaded / total * 100), 100)
                            progress.progress(pct, text=f"Downloading... {pct}% ({downloaded/1e6:.1f} MB)")

            progress.empty()
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.success(f"✅ Model downloaded ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            st.error(f"❌ Download error: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            return False


@st.cache_resource
def load_model():
    """Download and load model with Keras version compatibility."""

    if not download_model():
        return None

    # Try multiple loading strategies for Keras 2/3 compatibility
    model = None

    # Strategy 1: Standard load
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded (standard)")
        return model
    except Exception as e1:
        st.warning(f"Standard load failed: {e1}")

    # Strategy 2: Load with compile=False
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded (compile=False)")
        return model
    except Exception as e2:
        st.warning(f"compile=False load failed: {e2}")

    # Strategy 3: Use h5py + custom_objects
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=None
        )
        st.success("✅ Model loaded (custom_objects)")
        return model
    except Exception as e3:
        st.warning(f"custom_objects load failed: {e3}")

    # Strategy 4: Legacy h5 loader
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            model = keras_load(MODEL_PATH, compile=False)
        st.success("✅ Model loaded (h5py)")
        return model
    except Exception as e4:
        st.error(f"❌ All loading strategies failed.")
        st.error(f"Last error: {e4}")

        # Delete corrupted file so next reload tries fresh download
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None


@st.cache_data
def load_class_names():
    """Load class index mapping — checks multiple paths."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'models', 'class_indices.json'),
        os.path.join(os.path.dirname(__file__), '..', 'class_indices.json'),
        os.path.join(os.path.dirname(__file__), 'class_indices.json'),
        "models/class_indices.json",
        "class_indices.json",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                class_indices = json.load(f)
            return {v: k for k, v in class_indices.items()}

    st.error("❌ class_indices.json not found. Checked paths: " + str(possible_paths))
    return {}


# ================== ADVICE MAPS ==================

PEST_ADVICE_MAP = {
    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple___Apple_scab":                           "🍎 Use sulfur spray. Prune infected branches. Apply Captan before rain.",
    "Apple___Black_rot":                            "🍎 Remove mummified fruits. Apply copper-based fungicide every 10 days.",
    "Apple___Cedar_apple_rust":                     "🍎 Apply Myclobutanil fungicide at bud break. Remove nearby cedar trees.",
    "Apple___healthy":                              "✅ Apple plant is healthy. Maintain regular watering and pruning.",

    # ── Blueberry ──────────────────────────────────────────────────────────
    "Blueberry___healthy":                          "✅ Blueberry plant is healthy. Ensure acidic soil (pH 4.5–5.5).",

    # ── Cherry ─────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew":     "🍒 Apply sulfur-based fungicide. Improve air circulation between plants.",
    "Cherry_(including_sour)___healthy":            "✅ Cherry plant is healthy. Continue good orchard hygiene.",

    # ── Corn ───────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "🌽 Apply Azoxystrobin fungicide. Rotate crops annually.",
    "Corn_(maize)___Common_rust_":                  "🌽 Apply Propiconazole fungicide. Plant resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight":          "🌽 Use Mancozeb fungicide. Ensure proper field drainage.",
    "Corn_(maize)___healthy":                       "✅ Corn plant is healthy. Maintain adequate nitrogen fertilization.",

    # ── Grape ──────────────────────────────────────────────────────────────
    "Grape___Black_rot":                            "🍇 Remove infected berries. Apply Mancozeb or Captan fungicide.",
    "Grape___Esca_(Black_Measles)":                 "🍇 Prune infected wood. Use wound protection paste.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":   "🍇 Apply copper-based fungicide. Remove fallen infected leaves.",
    "Grape___healthy":                              "✅ Grape vine is healthy. Maintain proper canopy management.",

    # ── Orange ─────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)":     "🍊 No cure. Remove infected trees. Control Asian citrus psyllid with insecticide.",

    # ── Peach ──────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot":                       "🍑 Apply copper bactericide spray early season. Avoid overhead irrigation.",
    "Peach___healthy":                              "✅ Peach plant is healthy. Thin fruits to improve air circulation.",

    # ── Pepper ─────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot":                "🫑 Use copper-based bactericide. Rotate crops. Avoid wet conditions.",
    "Pepper,_bell___healthy":                       "✅ Bell pepper is healthy. Ensure consistent watering.",

    # ── Potato ─────────────────────────────────────────────────────────────
    "Potato___Early_blight":                        "🥔 Apply Chlorothalonil or Mancozeb fungicide. Remove lower infected leaves.",
    "Potato___Late_blight":                         "🥔 Apply Metalaxyl fungicide immediately. Destroy infected plants.",
    "Potato___healthy":                             "✅ Potato plant is healthy. Hill soil around plants.",

    # ── Raspberry ──────────────────────────────────────────────────────────
    "Raspberry___healthy":                          "✅ Raspberry plant is healthy. Prune old canes after harvest.",

    # ── Soybean ────────────────────────────────────────────────────────────
    "Soybean___healthy":                            "✅ Soybean plant is healthy. Monitor for aphids and spider mites.",

    # ── Squash ─────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew":                      "🎃 Apply potassium bicarbonate or sulfur spray. Increase plant spacing.",

    # ── Strawberry ─────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch":                     "🍓 Remove infected leaves. Apply Captan fungicide. Avoid overhead watering.",
    "Strawberry___healthy":                         "✅ Strawberry plant is healthy. Renew beds every 3–4 years.",

    # ── Tomato ─────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot":                      "🍅 Apply copper-based bactericide. Use disease-free seeds.",
    "Tomato___Early_blight":                        "🍅 Apply Chlorothalonil. Remove lower infected leaves. Mulch base.",
    "Tomato___Late_blight":                         "🍅 Apply copper-based fungicide. Remove infected leaves.",
    "Tomato___Leaf_Mold":                           "🍅 Improve greenhouse ventilation. Apply Mancozeb.",
    "Tomato___Septoria_leaf_spot":                  "🍅 Remove lower leaves. Apply Chlorothalonil every 7–10 days.",
    "Tomato___Spider_mites Two-spotted_spider_mite":"🍅 Apply Neem oil or Abamectin miticide. Increase humidity.",
    "Tomato___Target_Spot":                         "🍅 Apply Azoxystrobin fungicide. Ensure good air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":       "🍅 Control whitefly with insecticide. Remove infected plants.",
    "Tomato___Tomato_mosaic_virus":                 "🍅 No cure. Remove infected plants. Disinfect tools with bleach.",
    "Tomato___healthy":                             "✅ Tomato plant is healthy. Stake plants and prune suckers regularly.",
}


def get_pest_advice(disease_label: str) -> str:
    if disease_label in PEST_ADVICE_MAP:
        return PEST_ADVICE_MAP[disease_label]
    for key, advice in PEST_ADVICE_MAP.items():
        if key.lower() in disease_label.lower() or disease_label.lower() in key.lower():
            return advice
    if "healthy" in disease_label.lower():
        return "✅ Plant is healthy! Maintain good watering, fertilization, and monitoring."
    return "⚠️ Consult an agronomist for this disease. Avoid chemicals without proper diagnosis."


def preprocess_image(img: Image.Image) -> np.ndarray:
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ================== MAIN PAGE ==================

def pest_detection_page():
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload a **leaf image** to detect diseases using AI (**98.71% accuracy model**).")

    model = load_model()
    if model is None:
        st.error("❌ Model could not be loaded. Please check logs via 'Manage App' → Logs.")
        st.info("💡 Try refreshing the page. If error persists, the model file on HuggingFace may need to be re-uploaded.")
        st.stop()

    class_names = load_class_names()
    if not class_names:
        st.error("❌ Class names could not be loaded. Make sure class_indices.json is in your repo.")
        st.stop()

    st.success(f"✅ Model ready | {len(class_names)} disease classes loaded")
    st.markdown("---")

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

            with st.spinner("🔬 Analyzing leaf..."):
                preds = model.predict(x)
                probs = preds[0]
                idx = int(np.argmax(probs))
                confidence = float(np.max(probs) * 100.0)
                class_name = class_names[idx]

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
            st.info(get_pest_advice(class_name))

            st.markdown("---")
            st.subheader("📊 Top 3 Predictions")
            top3_idx = np.argsort(probs)[::-1][:3]
            for rank, i in enumerate(top3_idx, start=1):
                cname = class_names[int(i)].replace("___", " → ").replace("_", " ")
                prob = float(probs[int(i)] * 100.0)
                st.progress(int(prob), text=f"{rank}. {cname}: {prob:.1f}%")

            st.markdown("---")
            st.subheader("🌾 Detailed Advice for Top Predictions")
            for rank, i in enumerate(top3_idx, start=1):
                raw_label = class_names[int(i)]
                prob = float(probs[int(i)] * 100.0)
                display = raw_label.replace("___", " → ").replace("_", " ")
                with st.expander(f"{rank}. {display} ({prob:.1f}%)"):
                    st.write(get_pest_advice(raw_label))