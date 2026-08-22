import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

try:
    from database.db import get_all_farmers, add_pest_log
    HAS_DB = True
except Exception:
    HAS_DB = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

PEST_SOLUTIONS = {
    "Tomato___Early_blight": {
        "pest_name": "Tomato Early Blight", "type": "Fungal Disease", "severity": "High",
        "symptoms": "Dark brown spots with concentric rings (target-board pattern) on older leaves",
        "organic_solutions": [
            "Spray neem oil (5ml/liter) every 7 days",
            "Apply copper oxychloride 50% WP",
            "Remove infected lower leaves immediately",
            "Use mulch to prevent soil splash"
        ],
        "chemical_solutions": [
            "Mancozeb 75% WP @ 2.5 g/liter",
            "Chlorothalonil 75% WP @ 2 g/liter",
            "Azoxystrobin 23% SC @ 1 ml/liter"
        ],
        "prevention": [
            "Rotate crops every 3 years",
            "Stake plants for air circulation",
            "Water at base, avoid wetting leaves",
            "Use disease-free seeds"
        ],
        "best_season": "Monsoon & Post-monsoon"
    },
    "Tomato___Late_blight": {
        "pest_name": "Tomato Late Blight", "type": "Fungal Disease", "severity": "Very High",
        "symptoms": "Water-soaked lesions turning brown-black on leaves and stems, white fuzzy growth",
        "organic_solutions": [
            "Copper-based Bordeaux mixture (1%)",
            "Remove and destroy all infected material",
            "Improve drainage and airflow",
            "Biofungicide with Bacillus subtilis"
        ],
        "chemical_solutions": [
            "Metalaxyl + Mancozeb @ 2.5 g/liter",
            "Cymoxanil 8% + Mancozeb 64% WP @ 2.5 g/liter",
            "Dimethomorph 50% WP @ 1 g/liter"
        ],
        "prevention": [
            "Plant certified disease-free transplants",
            "Avoid dense planting",
            "Monitor weather forecasts for blight conditions",
            "Destroy crop debris after harvest"
        ],
        "best_season": "Cool, wet weather monitoring"
    },
    "default": {
        "pest_name": "Unknown Pest/Disease", "type": "Unknown", "severity": "Unknown",
        "symptoms": "Detected anomaly in plant health",
        "organic_solutions": [
            "Consult local agricultural extension officer",
            "Apply neem oil as general pesticide (5ml/liter)",
            "Remove visibly infected plant parts",
            "Improve drainage and air circulation"
        ],
        "chemical_solutions": [
            "Consult agronomist before chemical application",
            "General fungicide: Mancozeb 75% WP @ 2.5 g/liter",
            "General insecticide: Imidacloprid 17.8% SL @ 0.5 ml/liter"
        ],
        "prevention": [
            "Regular field monitoring",
            "Crop rotation",
            "Use certified seeds",
            "Maintain field hygiene"
        ],
        "best_season": "Year-round monitoring"
    }
}


def get_solution(class_name):
    if class_name in PEST_SOLUTIONS:
        return PEST_SOLUTIONS[class_name]
    for key in PEST_SOLUTIONS:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return PEST_SOLUTIONS[key]
    solution = PEST_SOLUTIONS["default"].copy()
    solution["pest_name"] = class_name.replace("___", " - ").replace("_", " ")
    return solution


@st.cache_resource(show_spinner=False)
def load_model():
    import json
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model not found at {MODEL_PATH}"
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        idx_to_class = {}
        if os.path.exists(INDICES_PATH):
            with open(INDICES_PATH, "r") as f:
                idx_to_class = {int(k): v for k, v in json.load(f).items()}
        return model, idx_to_class, None
    except Exception as e:
        return None, None, str(e)


def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_disease(model, img_array, idx_to_class):
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0])) * 100
    class_name = idx_to_class.get(idx, f"Class_{idx}")
    return class_name, confidence


def display_severity_badge(severity):
    colors = {"None": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Very High": "🔴🔴"}
    return colors.get(severity, "⚪")


def pest_detection_page():
    st.markdown("## 🔍 Pest & Plant Disease Detection")
    st.markdown("Upload a **crop or leaf image** — AI will detect **pests and diseases** and give specific treatment & prevention advice.")
    st.markdown("---")

    model, idx_to_class, error = load_model()

    if model is None:
        st.error(f"❌ Failed to load AI model: {error}")
        return

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

    col1, col2 = st.columns(2)
    image = None
    solution = None

    with col1:
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant/leaf image", type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the affected plant leaf"
        )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="📷 Uploaded Image", use_container_width=True)
                st.info(f"📊 Image size: {image.size[0]}x{image.size[1]} pixels")
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                return

    with col2:
        st.markdown("### 🔬 AI Diagnosis")

        if uploaded_file is None:
            st.info("👆 Upload an image on the left to begin.")
            st.markdown("### 📋 Model Info")
            st.markdown("- **Classes:** 38 plant diseases")
            st.markdown("- **Architecture:** MobileNetV2")
            st.markdown("- **Input:** 224 × 224 px")
        else:
            with st.spinner("🤖 Analyzing image with AI..."):
                try:
                    img_array = preprocess_image(image)
                    class_name, confidence = predict_disease(model, img_array, idx_to_class)
                    solution = get_solution(class_name)
                    display_name = solution["pest_name"]
                    severity_icon = display_severity_badge(solution["severity"])

                    st.markdown(f"### {severity_icon} {display_name}")

                    conf_color = "🟢" if confidence > 80 else "🟡" if confidence > 60 else "🔴"
                    st.markdown(f"**{conf_color} Confidence: {confidence:.1f}%**")
                    st.progress(confidence / 100)

                    if confidence < 60:
                        st.warning("⚠️ Low confidence. Please upload a clearer image.")

                    col_a, col_b = st.columns(2)
                    col_a.metric("Type", solution["type"])
                    col_b.metric("Severity", solution["severity"])

                    st.markdown(f"**🔍 Symptoms:** {solution['symptoms']}")

                    if HAS_DB:
                        try:
                            add_pest_log(selected_farmer_id, uploaded_file.name, class_name, confidence / 100)
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    solution = None

    if solution:
        st.markdown("---")
        st.markdown("## 💊 Treatment & Prevention Solutions")

        tab1, tab2, tab3 = st.tabs(["🌿 Organic Solutions", "🧪 Chemical Solutions", "🛡️ Prevention"])

        with tab1:
            for i, sol in enumerate(solution["organic_solutions"], 1):
                st.markdown(f"**{i}.** ✅ {sol}")

        with tab2:
            st.warning("⚠️ Always follow label instructions. Wear protective equipment.")
            for i, sol in enumerate(solution["chemical_solutions"], 1):
                st.markdown(f"**{i}.** 🧴 {sol}")

        with tab3:
            for i, prev in enumerate(solution["prevention"], 1):
                st.markdown(f"**{i}.** 🔰 {prev}")
            if "best_season" in solution:
                st.info(f"📅 **Best Monitoring Season:** {solution['best_season']}")

        if solution["severity"] in ["High", "Very High"]:
            st.error(
                "🚨 **URGENT ACTION REQUIRED!**\n\n"
                "- Begin treatment within 24-48 hours\n"
                "- Isolate heavily infected plants immediately\n"
                "- Contact local agricultural extension: **1800-180-1551** (India Toll-Free)"
            )