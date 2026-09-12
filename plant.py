"""
Plant Disease Detection Module
Loads a MobileNetV2-based Keras model and classifies 38 PlantVillage classes.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

try:
    from database.db import add_pest_log, get_all_farmers
    HAS_DB = True
except Exception:
    HAS_DB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")

INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

PLANT_MODEL_URL = "https://huggingface.co/bhaskar06/agri-plant-disease-model/resolve/main/plant_disease_model.h5"
PLANT_INDICES_URL = "https://huggingface.co/bhaskar06/agri-plant-disease-model/resolve/main/class_indices.json"


def _download_if_missing(path, url):
    if os.path.exists(path):
        return True
    try:
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return False


@st.cache_resource(show_spinner="Downloading and loading AI model (first run only)...")
def load_model():
    _download_if_missing(MODEL_PATH, PLANT_MODEL_URL)
    _download_if_missing(INDICES_PATH, PLANT_INDICES_URL)
    global CLASS_NAMES
    CLASS_NAMES = _load_class_names()

    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at {MODEL_PATH}"

    try:
        return tf.keras.models.load_model(MODEL_PATH), None
    except Exception:
        pass

    try:
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m, None
    except Exception:
        pass

    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
        m = tf.keras.Model(inputs=base.input, outputs=out)
        m.load_weights(MODEL_PATH, by_name=False, skip_mismatch=True)
        return m, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image, model):
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(preds[i])) for i in top3_idx]
    return CLASS_NAMES[top3_idx[0]], float(preds[top3_idx[0]]), top3


def pest_detection_page():
    st.title("🔍 Pest & Plant Disease Detection")
    st.markdown(
        "Upload a **crop or leaf image** — AI will detect **pests and diseases** "
        "and give specific treatment & prevention advice."
    )
    st.markdown("---")

    with st.spinner("Loading AI model..."):
        model, load_error = load_model()

    if model is None:
        st.error(f"❌ Could not load model: {load_error}")
        st.warning("Make sure `models/plant_disease_model.h5` exists in your repo, "
                    "and that it was saved with `model.build()` called before `model.save()`.")
        st.stop()

    st.success("✅ AI Model loaded successfully")

    selected_farmer_id = None
    if HAS_DB:
        try:
            farmers = get_all_farmers()
            if farmers:
                farmer_map = {f"[{f['id']}] {f['name']}": f['id'] for f in farmers}
                sel = st.selectbox("👨‍🌾 Link to Farmer (optional)", ["-- Skip --"] + list(farmer_map.keys()))
                if sel != "-- Skip --":
                    selected_farmer_id = farmer_map[sel]
        except Exception:
            pass

    st.markdown("---")
    col1, col2 = st.columns(2)
    image = None

    with col1:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose a leaf/crop image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔬 AI Diagnosis")

        if not uploaded:
            st.info("👈 Upload an image on the left to begin.")
            st.markdown("#### 📋 Model Info")
            st.write(f"- **Classes:** {len(CLASS_NAMES)} plant diseases")
            st.write("- **Architecture:** MobileNetV2")
            st.write("- **Input:** 224 × 224 px")
            with st.expander("📂 All Supported Classes"):
                for i, name in enumerate(CLASS_NAMES):
                    st.write(f"`{i}` — {name.replace('___', ' → ').replace('_', ' ')}")
            return

        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    disease, confidence, top3 = predict(image, model)
                    info = DISEASE_INFO.get(disease)
                    display_name = disease.replace("___", " → ").replace("_", " ").title()
                    conf_pct = confidence * 100

                    if conf_pct < 40:
                        st.error(
                            "⚠️ **Could not confidently identify this plant.** "
                            "This model only recognizes the crops listed below. If your "
                            "photo is of a different plant (e.g. mango, pomegranate), "
                            "results will not be reliable."
                        )
                    elif "healthy" in disease.lower():
                        st.success(f"✅ **Diagnosis:** {display_name}")
                    elif conf_pct >= 70:
                        st.error(f"⚠️ **Diagnosis:** {display_name}")
                    else:
                        st.warning(f"🟡 **Diagnosis:** {display_name} *(Moderate confidence)*")

                    st.metric("Confidence", f"{conf_pct:.1f}%")
                    st.progress(min(confidence, 1.0))
                    st.markdown("---")

                    st.markdown("### 📊 Top 3 Predictions")
                    for rank, (cls, conf) in enumerate(top3):
                        label = cls.replace("___", " → ").replace("_", " ").title()
                        st.write(f"**{rank + 1}.** {label} — `{conf * 100:.1f}%`")
                        st.progress(min(conf, 1.0))

                    st.markdown("---")

                    if info:
                        sev = info.get("severity", "Unknown")
                        sev_emoji = {"None": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🆘"}.get(sev, "⚪")
                        st.markdown("### 💊 Treatment & Prevention")
                        ca, cb = st.columns(2)
                        ca.markdown(f"**Type:** {info['type']}")
                        cb.markdown(f"**Severity:** {sev_emoji} {sev}")

                        if "healthy" in disease.lower():
                            st.success(f"✅ {info['treatment']}")
                            st.info(f"**Prevention:** {info['prevention']}")
                        else:
                            st.error(f"**🚨 Treatment:** {info['treatment']}")
                            st.warning(f"**🛡️ Prevention:** {info['prevention']}")

                        tab1, tab2 = st.tabs(["🧪 Chemical Pesticide", "🌿 Organic Alternative"])
                        with tab1:
                            st.write(info['pesticide'])
                            st.caption("⚠️ Wear protective equipment. Follow label instructions.")
                        with tab2:
                            st.write(info['organic'])

                        if sev in ["High", "Critical"]:
                            st.error(
                                "🚨 **URGENT ACTION REQUIRED!**\n\n"
                                "- Begin treatment within 24–48 hours\n"
                                "- Isolate heavily infected plants immediately\n"
                                "- Contact local agriculture helpline: **1800-180-1551** (India Toll-Free)"
                            )
                    else:
                        st.info("ℹ️ Consult your local agricultural extension officer for targeted treatment.")

                    if HAS_DB:
                        try:
                            solution = info['treatment'] if info else "Consult agronomist"
                            add_pest_log(selected_farmer_id, uploaded.name, disease, confidence, solution)
                            if selected_farmer_id:
                                st.caption("✅ Saved to farmer profile.")
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("Ensure the image is a clear photo of a plant leaf.")