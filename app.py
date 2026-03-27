import streamlit as st
import os

st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database.db import init_db, get_connection
from modules.farmer_management import farmer_management_page
from modules.crop_management import crop_management_page
from modules.pest_detection import pest_detection_page
from modules.soil_health import soil_health_page

import tensorflow as tf
import numpy as np
import json
from PIL import Image

init_db()

# ── Plant Disease Model Loader ──────────────────────────────────────────────
@st.cache_resource
def load_plant_model():
    model_path = "models/plant_disease_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_data
def load_class_indices():
    path = "models/class_indices.json"
    if os.path.exists(path):
        with open(path) as f:
            indices = json.load(f)
        return {v: k for k, v in indices.items()}
    return {}

# ── Session State ────────────────────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Dashboard"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 Smart Agriculture")
    st.markdown("**AI-Powered Farm Management**")
    st.markdown("---")
    pages = [
        "🏠 Dashboard",
        "👨‍🌾 Farmer Management",
        "🌱 Crop Management",
        "🔍 Pest Detection",
        "🌿 Plant Disease Detection",
        "🧪 Soil Health"
    ]
    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state["active_page"] = p
    st.markdown("---")
    st.caption("© 2026 AI Smart Agriculture")

# ── Top Nav ──────────────────────────────────────────────────────────────────
st.markdown("### 🌾 AI Smart Agriculture")
cols = st.columns(6)
nav_labels = ["🏠 Dashboard", "👨‍🌾 Farmers", "🌱 Crops", "🔍 Pest Detection", "🌿 Plant Disease", "🧪 Soil Health"]
nav_pages  = ["🏠 Dashboard", "👨‍🌾 Farmer Management", "🌱 Crop Management", "🔍 Pest Detection", "🌿 Plant Disease Detection", "🧪 Soil Health"]
for i, col in enumerate(cols):
    with col:
        if st.button(nav_labels[i], key=f"top_{i}", use_container_width=True):
            st.session_state["active_page"] = nav_pages[i]

st.markdown("---")
active_page = st.session_state["active_page"]

# ── Pages ─────────────────────────────────────────────────────────────────────

if active_page == "🏠 Dashboard":
    st.title("🌾 AI Smart Agriculture Dashboard")
    st.markdown("""
    ### Welcome to your Smart Farm Management System
    Use the **sidebar** or **buttons above** to navigate:
    - **👨‍🌾 Farmer Management** — Register and manage farmer profiles
    - **🌱 Crop Management** — Track planting schedules + get pest protection advice
    - **🔍 Pest Detection** — Upload leaf images for AI diagnosis
    - **🌿 Plant Disease Detection** — Deep learning model to identify plant diseases
    - **🧪 Soil Health** — Get crop & fertilizer recommendations
    """)
    conn = get_connection()
    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Farmers",    conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0])
    c2.metric("🌱 Crop Records", conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0])
    c3.metric("🧪 Soil Records", conn.execute("SELECT COUNT(*) FROM soil_records").fetchone()[0])
    conn.close()

elif active_page == "👨‍🌾 Farmer Management":
    farmer_management_page()

elif active_page == "🌱 Crop Management":
    crop_management_page()

elif active_page == "🔍 Pest Detection":
    pest_detection_page()

elif active_page == "🌿 Plant Disease Detection":
    st.title("🌿 Plant Disease Detection")
    st.markdown("Upload a **leaf image** and the AI model will diagnose the disease with confidence score.")
    st.markdown("---")

    model = load_plant_model()
    idx_to_class = load_class_indices()

    if model is None:
        st.error("❌ Model not found at `models/plant_disease_model.h5`. Please train the model first.")
        st.stop()

    if not idx_to_class:
        st.error("❌ Class indices not found at `models/class_indices.json`. Please train the model first.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Leaf", use_column_width=True)

    with col2:
        st.subheader("🔬 Diagnosis Result")

        if uploaded_file:
            with st.spinner("Analyzing image..."):
                # Preprocess
                img_resized = img.resize((224, 224))
                arr = np.array(img_resized) / 255.0
                arr = np.expand_dims(arr, axis=0)

                # Predict
                preds = model.predict(arr)
                idx = int(np.argmax(preds))
                confidence = float(preds[0][idx]) * 100
                label = idx_to_class.get(idx, "Unknown")

                # Clean up label for display
                display_label = label.replace("_", " ").title()

            # Results
            if confidence >= 80:
                st.success(f"**🌿 Diagnosis:** {display_label}")
            elif confidence >= 50:
                st.warning(f"**🌿 Diagnosis:** {display_label}")
            else:
                st.error(f"**🌿 Diagnosis:** {display_label} (Low confidence)")

            st.metric("Confidence Score", f"{confidence:.2f}%")
            st.progress(min(confidence / 100, 1.0))

            st.markdown("---")

            # Top 3 predictions
            st.subheader("📊 Top 3 Predictions")
            top3_idx = np.argsort(preds[0])[::-1][:3]
            for rank, i in enumerate(top3_idx):
                lbl = idx_to_class.get(int(i), "Unknown").replace("_", " ").title()
                conf = float(preds[0][i]) * 100
                st.write(f"**{rank+1}.** {lbl} — `{conf:.2f}%`")
                st.progress(min(conf / 100, 1.0))

            st.markdown("---")

            # Basic advice
            st.subheader("💡 General Advice")
            label_lower = label.lower()
            if "healthy" in label_lower:
                st.success("✅ Your plant looks healthy! Keep up good farming practices.")
            elif "blight" in label_lower:
                st.warning("⚠️ Blight detected. Apply copper-based fungicide and remove infected leaves immediately.")
            elif "rust" in label_lower:
                st.warning("⚠️ Rust detected. Use sulfur-based fungicide and ensure good air circulation.")
            elif "spot" in label_lower or "leaf" in label_lower:
                st.warning("⚠️ Leaf spot detected. Avoid overhead watering and apply appropriate fungicide.")
            elif "mold" in label_lower or "mildew" in label_lower:
                st.warning("⚠️ Mold/Mildew detected. Improve ventilation and apply anti-fungal treatment.")
            else:
                st.info("ℹ️ Disease detected. Please consult an agronomist for targeted treatment.")

        else:
            st.info("👈 Upload a leaf image on the left to get started.")

            st.markdown("#### 📋 Model Info")
            st.write(f"- **Total Classes:** {len(idx_to_class)}")
            st.write(f"- **Model:** Plant Disease CNN (98.81% val accuracy)")
            st.write(f"- **Input Size:** 224 × 224 px")

            with st.expander("🗂️ View All Disease Classes"):
                for idx, name in sorted(idx_to_class.items()):
                    st.write(f"`{idx}` — {name.replace('_', ' ').title()}")

elif active_page == "🧪 Soil Health":
    soil_health_page()