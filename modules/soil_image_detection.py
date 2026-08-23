import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

try:
    from database.db import get_all_farmers, db_add_soil_image_log, db_get_soil_image_logs
    HAS_DB = True
except Exception:
    HAS_DB = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "soil_type_model.h5")

CLASS_NAMES = ["Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil"]

# Facts and advice per soil type — grounded in soil science, not guessed from the photo.
# The photo tells us WHICH type of soil it is; everything below is known, established
# information about that type, not something extracted from the image itself.
SOIL_INFO = {
    "Alluvial Soil": {
        "description": "Formed by river deposits. Fertile, rich in potash, phosphoric acid and lime. "
                        "Most widespread and agriculturally important soil type.",
        "moisture_retention": "Good — retains moisture well while still draining excess water.",
        "typical_ph": "6.5 – 7.5 (near neutral)",
        "ideal_crops": ["Rice", "Wheat", "Sugarcane", "Maize", "Pulses", "Oilseeds"],
        "condition": "Good",
        "improvement_tips": [
            "Generally fertile — maintain with balanced NPK fertilization",
            "Add organic compost every season to sustain fertility",
            "Practice crop rotation to prevent nutrient depletion"
        ]
    },
    "Black Soil": {
        "description": "Also called 'Regur soil'. Rich in clay, high water retention, expands and cracks "
                        "when dry. Ideal for cotton cultivation.",
        "moisture_retention": "Very high — but can become waterlogged if drainage is poor.",
        "typical_ph": "7.0 – 8.5 (neutral to slightly alkaline)",
        "ideal_crops": ["Cotton", "Soybean", "Sugarcane", "Wheat", "Sunflower"],
        "condition": "Good",
        "improvement_tips": [
            "Ensure proper drainage channels to prevent waterlogging",
            "Deep ploughing before monsoon improves aeration",
            "Add gypsum if soil becomes too alkaline over time"
        ]
    },
    "Clay Soil": {
        "description": "Fine particles, poor drainage, retains water for long periods. "
                        "Can become compacted and hard when dry.",
        "moisture_retention": "Very high — often too high, leading to root suffocation.",
        "typical_ph": "Varies, often slightly acidic to neutral",
        "ideal_crops": ["Rice", "Broccoli", "Cabbage", "Leafy greens"],
        "condition": "Needs Improvement",
        "improvement_tips": [
            "⚠️ Add coarse sand and organic matter (compost/manure) to improve drainage",
            "⚠️ Avoid working the soil when wet — it compacts easily",
            "⚠️ Raised beds help prevent waterlogging of root zones",
            "⚠️ Add gypsum to improve soil structure over time"
        ]
    },
    "Red Soil": {
        "description": "Formed from weathering of ancient crystalline rocks. Red color from iron oxide. "
                        "Generally low in nitrogen, phosphorus and organic matter.",
        "moisture_retention": "Low — drains quickly, dries out fast, prone to erosion.",
        "typical_ph": "5.5 – 7.0 (slightly acidic)",
        "ideal_crops": ["Groundnut", "Millets", "Potato", "Pulses", "Tobacco"],
        "condition": "Needs Improvement",
        "improvement_tips": [
            "⚠️ Low fertility — apply nitrogen and phosphorus-rich fertilizers (Urea, DAP)",
            "⚠️ Add organic compost/farmyard manure to boost nutrient content",
            "⚠️ Mulching helps retain the limited moisture available",
            "⚠️ Consider drip irrigation since water drains away quickly"
        ]
    },
}


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found at {MODEL_PATH}. Run train_soil_image_model.py first."
    try:
        return tf.keras.models.load_model(MODEL_PATH), None
    except Exception:
        pass
    try:
        m = tf.keras.models.load_model(MODEL_PATH, compile=False)
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m, None
    except Exception as e:
        return None, str(e)


def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_soil(image, model):
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx])


def soil_image_detection_page():
    st.markdown("## 📷 Soil Type Detection (Image-based)")
    st.markdown(
        "Upload a **photo of soil** — AI identifies the soil type and gives crop "
        "suitability and improvement advice."
    )
    st.caption(
        "ℹ️ Note: a photo can reliably identify soil *type* (color/texture), which is what "
        "drives the recommendations below. Exact moisture %, pH, and nutrient levels still "
        "require a lab test or sensor — use the **Soil Input & Recommend** tab if you have "
        "those readings."
    )
    st.markdown("---")

    with st.spinner("Loading soil classification model..."):
        model, error = load_model()

    if model is None:
        st.error(f"❌ Could not load model: {error}")
        return

    st.success("✅ Soil AI Model loaded successfully")

    selected_farmer_id = None
    if HAS_DB:
        try:
            farmers = get_all_farmers()
            if farmers:
                farmer_map = {f"[{f['id']}] {f['name']}": f['id'] for f in farmers}
                sel = st.selectbox("👨‍🌾 Link to Farmer (optional)", ["-- Skip --"] + list(farmer_map.keys()), key="soilimg_farmer")
                if sel != "-- Skip --":
                    selected_farmer_id = farmer_map[sel]
        except Exception:
            pass

    col1, col2 = st.columns(2)
    image = None

    with col1:
        st.subheader("📤 Upload Soil Image")
        uploaded = st.file_uploader(
            "Choose a soil photo (JPG or PNG)", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed", key="soilimg_upload"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Soil Image", use_container_width=True)

    with col2:
        st.subheader("🔬 AI Diagnosis")

        if not uploaded:
            st.info("👈 Upload a soil image on the left to begin.")
            st.markdown("#### 📋 Model Info")
            st.write(f"- **Classes:** {len(CLASS_NAMES)} soil types")
            st.write("- **Architecture:** MobileNetV2 (fine-tuned)")
            st.write("- **Input:** 224 × 224 px")
            return

        if st.button("🔬 Analyze Soil", type="primary", use_container_width=True, key="soilimg_btn"):
            with st.spinner("Analyzing..."):
                try:
                    soil_type, confidence = predict_soil(image, model)
                    info = SOIL_INFO.get(soil_type)
                    conf_pct = confidence * 100

                    if conf_pct >= 70:
                        st.success(f"✅ **Detected Soil Type:** {soil_type}")
                    else:
                        st.warning(f"🔍 **Detected Soil Type:** {soil_type} *(Low confidence — try a clearer, well-lit photo)*")

                    st.metric("Confidence", f"{conf_pct:.1f}%")
                    st.progress(min(confidence, 1.0))
                    st.markdown("---")

                    if info:
                        cond_color = "🟢" if info["condition"] == "Good" else "🟡"
                        st.markdown(f"### {cond_color} Soil Condition: {info['condition']}")
                        st.write(info["description"])

                        ca, cb = st.columns(2)
                        ca.metric("Moisture Retention", info["moisture_retention"].split("—")[0].strip())
                        cb.metric("Typical pH Range", info["typical_ph"])

                        st.markdown("#### 🌾 Suitable Crops")
                        st.write(", ".join(info["ideal_crops"]))

                        st.markdown("#### 🛠️ Recommendations")
                        for tip in info["improvement_tips"]:
                            st.markdown(f"- {tip}")

                    if HAS_DB:
                        try:
                            db_add_soil_image_log(selected_farmer_id, uploaded.name, soil_type, confidence)
                            if selected_farmer_id:
                                st.caption("✅ Saved to farmer profile.")
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    if HAS_DB:
        st.markdown("---")
        with st.expander("📋 Past Soil Image Scans"):
            logs = db_get_soil_image_logs()
            if logs:
                import pandas as pd
                rows = [{
                    "Farmer": (r.get("farmers") or {}).get("name", "Guest"),
                    "Soil Type": r.get("soil_type", ""),
                    "Confidence": f"{r.get('confidence', 0) * 100:.1f}%",
                    "Date": str(r.get("created_at", ""))[:10],
                } for r in logs]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No soil image scans yet.")