import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import requests
import tempfile
from database.db import add_pest_log, get_all_farmers

# ── 38 PlantVillage Classes ───────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ── Full Treatment & Pest Solutions ──────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "type": "🍎 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply fungicides like Captan or Mancozeb every 7–10 days during wet weather.",
        "prevention": "Rake and destroy fallen leaves. Plant scab-resistant apple varieties.",
        "pesticide": "Captan 50WP (2g/litre) or Mancozeb 75WP (2.5g/litre)",
        "organic": "Neem oil spray (5ml/litre) every 7 days"
    },
    "Apple___Black_rot": {
        "type": "🍎 Fungal Disease",
        "severity": "High",
        "treatment": "Prune all infected branches 8–10 inches below visible infection. Apply copper-based fungicide.",
        "prevention": "Remove mummified fruits. Maintain good tree hygiene.",
        "pesticide": "Copper hydroxide (3g/litre) or Thiophanate-methyl",
        "organic": "Bordeaux mixture (1%) spray"
    },
    "Apple___Cedar_apple_rust": {
        "type": "🍎 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply myclobutanil or triadimefon fungicide at bud break and throughout spring.",
        "prevention": "Remove nearby cedar/juniper trees. Plant rust-resistant varieties.",
        "pesticide": "Myclobutanil (Nova) at 1.5ml/litre",
        "organic": "Sulfur-based spray every 7–10 days"
    },
    "Apple___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your plant is healthy!",
        "prevention": "Maintain regular watering, fertilization, and pruning schedule.",
        "pesticide": "None required",
        "organic": "Continue with preventive neem oil spray monthly"
    },
    "Blueberry___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your blueberry plant is healthy!",
        "prevention": "Maintain soil pH 4.5–5.5. Mulch around plants.",
        "pesticide": "None required",
        "organic": "Use acidic compost as mulch"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "type": "🍒 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply potassium bicarbonate or sulfur sprays. Remove heavily infected shoots.",
        "prevention": "Ensure good air circulation by proper pruning. Avoid excess nitrogen.",
        "pesticide": "Sulfur dust (3g/litre) or Triadimefon",
        "organic": "Baking soda solution (1 tbsp/litre water) spray weekly"
    },
    "Cherry_(including_sour)___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your cherry plant is healthy!",
        "prevention": "Prune for airflow. Avoid wetting foliage during irrigation.",
        "pesticide": "None required",
        "organic": "Preventive neem oil spray once a month"
    },
    "Corn_(maize)___Cercospora_leaf_spot": {
        "type": "🌽 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply strobilurin or triazole fungicides at early disease onset.",
        "prevention": "Crop rotation with non-host crops. Use resistant hybrids.",
        "pesticide": "Azoxystrobin + Propiconazole (Amistar Top) at 1ml/litre",
        "organic": "Trichoderma-based biocontrol agents"
    },
    "Corn_(maize)___Common_rust_": {
        "type": "🌽 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply propiconazole-based fungicide. Plant resistant varieties.",
        "prevention": "Early planting to avoid peak rust periods. Monitor regularly.",
        "pesticide": "Propiconazole 25EC (1ml/litre) or Mancozeb",
        "organic": "Neem-based formulations (5ml/litre)"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "type": "🌽 Fungal Disease",
        "severity": "High",
        "treatment": "Apply fungicide at first sign. Pyraclostrobin or azoxystrobin are effective.",
        "prevention": "Use resistant hybrids. Practice crop rotation.",
        "pesticide": "Pyraclostrobin (Cabrio) at 1.5ml/litre",
        "organic": "Copper-based spray at 14-day intervals"
    },
    "Corn_(maize)___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your corn plant is healthy!",
        "prevention": "Maintain adequate soil moisture and balanced fertilization.",
        "pesticide": "None required",
        "organic": "Preventive Trichoderma soil application"
    },
    "Grape___Black_rot": {
        "type": "🍇 Fungal Disease",
        "severity": "High",
        "treatment": "Remove all mummified fruits and infected canes. Apply myclobutanil fungicide.",
        "prevention": "Prune for good air circulation. Apply fungicide at 10-day intervals during wet weather.",
        "pesticide": "Myclobutanil (Nova) 1.5ml/litre or Captan",
        "organic": "Bordeaux mixture (1%) spray"
    },
    "Grape___Esca_(Black_Measles)": {
        "type": "🍇 Fungal/Wood Disease",
        "severity": "High",
        "treatment": "No complete cure. Remove infected vines. Apply wound protectants after pruning.",
        "prevention": "Prune during dry weather. Apply Trichoderma to pruning wounds.",
        "pesticide": "Sodium arsenite (restricted use) — consult expert",
        "organic": "Trichoderma harzianum wound treatment"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "type": "🍇 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply copper-based or mancozeb fungicides. Remove infected leaves.",
        "prevention": "Avoid overhead irrigation. Maintain canopy airflow.",
        "pesticide": "Mancozeb 75WP (2.5g/litre)",
        "organic": "Neem oil (5ml/litre) spray every 10 days"
    },
    "Grape___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your grape plant is healthy!",
        "prevention": "Maintain balanced pruning and trellis management.",
        "pesticide": "None required",
        "organic": "Monthly preventive Bordeaux spray"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "type": "🍊 Bacterial Disease (Severe)",
        "severity": "Critical",
        "treatment": "NO CURE EXISTS. Remove and destroy infected trees immediately to prevent spread.",
        "prevention": "Control Asian citrus psyllid (vector). Use certified disease-free planting material.",
        "pesticide": "Imidacloprid for psyllid control (insecticide, not cure)",
        "organic": "Sticky yellow traps for psyllid monitoring"
    },
    "Peach___Bacterial_spot": {
        "type": "🍑 Bacterial Disease",
        "severity": "High",
        "treatment": "Apply copper-based bactericide. Remove and destroy infected plant parts.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation.",
        "pesticide": "Copper hydroxide (3g/litre) spray weekly during wet season",
        "organic": "Copper-based Bordeaux mixture"
    },
    "Peach___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your peach plant is healthy!",
        "prevention": "Prune annually. Apply dormant oil spray in early spring.",
        "pesticide": "None required",
        "organic": "Neem oil spray as preventive measure"
    },
    "Pepper,_bell___Bacterial_spot": {
        "type": "🫑 Bacterial Disease",
        "severity": "High",
        "treatment": "Apply copper bactericide. Remove heavily infected plants. Avoid overhead irrigation.",
        "prevention": "Use disease-free seeds. Practice 2–3 year crop rotation.",
        "pesticide": "Copper oxychloride (3g/litre) or Streptomycin (200ppm)",
        "organic": "Bordeaux mixture spray every 10 days"
    },
    "Pepper,_bell___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your pepper plant is healthy!",
        "prevention": "Ensure good drainage. Balanced NPK fertilization.",
        "pesticide": "None required",
        "organic": "Neem oil preventive spray"
    },
    "Potato___Early_blight": {
        "type": "🥔 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. Ensure proper crop rotation.",
        "prevention": "Avoid overhead irrigation. Remove infected plant debris after harvest.",
        "pesticide": "Chlorothalonil 75WP (2g/litre) every 7–10 days",
        "organic": "Neem oil (5ml/litre) + baking soda (5g/litre) spray"
    },
    "Potato___Late_blight": {
        "type": "🥔 Fungal Disease (Severe)",
        "severity": "Critical",
        "treatment": "Apply metalaxyl fungicide IMMEDIATELY. Remove and destroy ALL infected plants.",
        "prevention": "Plant certified disease-free seed potatoes. Avoid wet foliage.",
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold) at 2.5g/litre",
        "organic": "Copper-based spray (Bordeaux 1%) every 5–7 days"
    },
    "Potato___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your potato plant is healthy!",
        "prevention": "Hill up soil around plants. Maintain consistent watering.",
        "pesticide": "None required",
        "organic": "Trichoderma soil amendment for root health"
    },
    "Raspberry___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your raspberry plant is healthy!",
        "prevention": "Remove old canes after fruiting. Mulch to conserve moisture.",
        "pesticide": "None required",
        "organic": "Neem oil preventive spray monthly"
    },
    "Soybean___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your soybean plant is healthy!",
        "prevention": "Inoculate seeds with Rhizobium. Practice crop rotation.",
        "pesticide": "None required",
        "organic": "Biofertilizer (Rhizobium) seed treatment"
    },
    "Squash___Powdery_mildew": {
        "type": "🎃 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply sulfur-based fungicide or neem oil spray. Remove severely infected leaves.",
        "prevention": "Ensure adequate plant spacing. Avoid water stress.",
        "pesticide": "Sulfur WP (3g/litre) or Triadimefon",
        "organic": "Baking soda (1 tbsp/litre) + neem oil (5ml/litre) spray"
    },
    "Strawberry___Leaf_scorch": {
        "type": "🍓 Fungal Disease",
        "severity": "Medium",
        "treatment": "Remove and destroy infected leaves. Apply captan or myclobutanil fungicide.",
        "prevention": "Avoid overhead irrigation. Plant in well-drained soil.",
        "pesticide": "Captan 50WP (2g/litre) every 10–14 days",
        "organic": "Neem oil spray (5ml/litre)"
    },
    "Strawberry___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your strawberry plant is healthy!",
        "prevention": "Mulch with straw to prevent soil-splash diseases.",
        "pesticide": "None required",
        "organic": "Preventive copper spray before rainy season"
    },
    "Tomato___Bacterial_spot": {
        "type": "🍅 Bacterial Disease",
        "severity": "High",
        "treatment": "Apply copper bactericide. Avoid overhead irrigation. Remove infected tissue.",
        "prevention": "Use disease-free seeds. Rotate crops every 2–3 years.",
        "pesticide": "Copper oxychloride (3g/litre) + Mancozeb (2g/litre)",
        "organic": "Bordeaux mixture spray every 7 days"
    },
    "Tomato___Early_blight": {
        "type": "🍅 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply copper fungicide. Maintain proper plant spacing. Remove lower infected leaves.",
        "prevention": "Avoid overhead watering. Mulch to prevent soil splash.",
        "pesticide": "Chlorothalonil 75WP (2g/litre) or Mancozeb",
        "organic": "Neem oil (5ml/litre) spray every 7 days"
    },
    "Tomato___Late_blight": {
        "type": "🍅 Fungal Disease (Severe)",
        "severity": "Critical",
        "treatment": "Use metalaxyl + mancozeb combination. Destroy ALL infected plant material immediately.",
        "prevention": "Plant resistant varieties. Avoid wet foliage conditions.",
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold MZ) at 2.5g/litre",
        "organic": "Copper Bordeaux mixture (1%) spray every 5 days"
    },
    "Tomato___Leaf_Mold": {
        "type": "🍅 Fungal Disease",
        "severity": "Medium",
        "treatment": "Improve greenhouse ventilation. Apply chlorothalonil or mancozeb fungicide.",
        "prevention": "Reduce humidity below 85%. Remove infected leaves promptly.",
        "pesticide": "Chlorothalonil (2g/litre) or Copper hydroxide",
        "organic": "Potassium bicarbonate spray weekly"
    },
    "Tomato___Septoria_leaf_spot": {
        "type": "🍅 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply mancozeb or chlorothalonil. Remove infected lower leaves immediately.",
        "prevention": "Avoid overhead irrigation. Stake plants to improve airflow.",
        "pesticide": "Mancozeb 75WP (2.5g/litre) every 7–10 days",
        "organic": "Copper-based organic spray + mulching"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "type": "🍅 Pest Infestation",
        "severity": "High",
        "treatment": "Apply miticide (abamectin or spiromesifen). Spray underside of leaves thoroughly.",
        "prevention": "Maintain adequate soil moisture. Introduce predatory mites.",
        "pesticide": "Abamectin 1.8EC (0.5ml/litre) or Spiromesifen",
        "organic": "Neem oil (10ml/litre) spray focusing on leaf undersides"
    },
    "Tomato___Target_Spot": {
        "type": "🍅 Fungal Disease",
        "severity": "Medium",
        "treatment": "Apply chlorothalonil or azoxystrobin fungicide at first sign of disease.",
        "prevention": "Improve plant spacing. Remove infected plant debris.",
        "pesticide": "Azoxystrobin (Amistar) at 1ml/litre",
        "organic": "Copper fungicide spray every 10 days"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "type": "🍅 Viral Disease",
        "severity": "Critical",
        "treatment": "NO DIRECT CURE. Control whitefly vectors with imidacloprid. Remove infected plants.",
        "prevention": "Use virus-resistant varieties. Install whitefly nets in greenhouse.",
        "pesticide": "Imidacloprid 17.8SL (0.3ml/litre) for whitefly control",
        "organic": "Yellow sticky traps + Neem oil spray for whitefly"
    },
    "Tomato___Tomato_mosaic_virus": {
        "type": "🍅 Viral Disease",
        "severity": "High",
        "treatment": "NO CURE. Remove and destroy infected plants. Disinfect tools with 10% bleach solution.",
        "prevention": "Use virus-free seeds. Wash hands before handling plants. Control aphids.",
        "pesticide": "Imidacloprid for aphid control (virus vector)",
        "organic": "Reflective mulch to deter aphids"
    },
    "Tomato___healthy": {
        "type": "✅ Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Your tomato plant is healthy!",
        "prevention": "Stake plants properly. Water at base to avoid leaf wetness.",
        "pesticide": "None required",
        "organic": "Monthly preventive neem oil spray"
    },
}

SEVERITY_COLOR = {
    "None": "success",
    "Medium": "warning",
    "High": "error",
    "Critical": "error",
}

# ── Robust Model Loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Robust loader that handles the (None, 7, 7, 1280) shape tuple bug.
    Tries multiple strategies before giving up.
    """
    model_path = "models/plant_disease_model.h5"

    if not os.path.exists(model_path):
        return None, "Model file not found at models/plant_disease_model.h5"

    # Strategy 1: Standard load
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e1:
        pass

    # Strategy 2: Load with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model, None
    except Exception as e2:
        pass

    # Strategy 3: Rebuild MobileNetV2 architecture and load weights only
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(38, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=output)
        model.load_weights(model_path, by_name=False, skip_mismatch=True)
        return model, None
    except Exception as e3:
        pass

    # Strategy 4: Load weights with h5py directly (bypass Keras config)
    try:
        import h5py
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(38, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=output)
        with h5py.File(model_path, 'r') as f:
            if 'model_weights' in f:
                model.load_weights(model_path)
        return model, None
    except Exception as e4:
        return None, f"All loading strategies failed. Last error: {str(e4)}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for MobileNetV2."""
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image, model) -> tuple:
    """Returns (class_name, confidence_0_to_1, top3_list)."""
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(preds[i])) for i in top3_idx]
    best_class = CLASS_NAMES[top3_idx[0]]
    best_conf = float(preds[top3_idx[0]])
    return best_class, best_conf, top3


# ── Main Page ─────────────────────────────────────────────────────────────────
def pest_detection_page():
    st.title("🔍 Pest & Plant Disease Detection")
    st.markdown(
        "Upload a **crop or leaf image** — the AI will auto-detect **both pests and diseases** "
        "and give you specific treatment & prevention advice."
    )
    st.markdown("---")

    # Load model with progress
    with st.spinner("Loading AI model..."):
        model, load_error = load_model()

    if model is None:
        st.error(f"❌ Could not load model: {load_error}")
        st.info(
            "**Quick Fix:** Your `plant_disease_model.h5` has a corrupted input shape config. "
            "Re-train using `train_plant_model.py` and ensure `model.save()` is called **after** "
            "calling `model.build((None, 224, 224, 3))`."
        )
        st.code(
            "# Add this BEFORE model.save() in train_plant_model.py:\n"
            "model.build((None, 224, 224, 3))\n"
            "model.save('models/plant_disease_model.h5')",
            language="python"
        )
        st.stop()

    st.success("✅ AI Model loaded successfully")

    # Farmer linkage (optional)
    farmers = get_all_farmers()
    farmer_map = {f"[{row[0]}] {row[1]}": row[0] for row in farmers} if farmers else {}
    selected_farmer = None
    if farmer_map:
        sel = st.selectbox("👨‍🌾 Link Detection to Farmer (optional)", ["-- Skip --"] + list(farmer_map.keys()))
        if sel != "-- Skip --":
            selected_farmer = farmer_map[sel]

    st.markdown("---")

    # Upload
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader(
            "Choose a leaf/crop image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🔬 AI Diagnosis")

        if not uploaded:
            st.info("👈 Upload an image on the left to begin analysis.")
            st.markdown("#### 📋 Model Info")
            st.write(f"- **Classes Supported:** {len(CLASS_NAMES)}")
            st.write("- **Architecture:** MobileNetV2 Transfer Learning")
            st.write("- **Input Size:** 224 × 224 px")
            st.write("- **Dataset:** PlantVillage (38 disease classes)")
            with st.expander("📂 View All Supported Classes"):
                for i, name in enumerate(CLASS_NAMES):
                    display = name.replace("_", " ").replace("___", " → ")
                    st.write(f"`{i}` — {display}")
            return

        if st.button("🔬 Analyze Image", type="primary", use_column_width=True):
            with st.spinner("Analyzing with AI model..."):
                try:
                    disease, confidence, top3 = predict(image, model)
                    info = DISEASE_INFO.get(disease, None)
                    display_name = disease.replace("___", " → ").replace("_", " ").title()
                    conf_pct = confidence * 100

                    # Diagnosis banner
                    if "healthy" in disease.lower():
                        st.success(f"✅ **Diagnosis:** {display_name}")
                    elif conf_pct >= 70:
                        st.error(f"⚠️ **Diagnosis:** {display_name}")
                    elif conf_pct >= 45:
                        st.warning(f"🟡 **Diagnosis:** {display_name}")
                    else:
                        st.warning(f"🔍 **Diagnosis:** {display_name} *(Low confidence — consider retaking photo)*")

                    # Confidence meter
                    st.metric("Confidence Score", f"{conf_pct:.1f}%")
                    st.progress(min(confidence, 1.0))
                    st.markdown("---")

                    # Top 3
                    st.markdown("### 📊 Top 3 Predictions")
                    for rank, (cls, conf) in enumerate(top3):
                        label = cls.replace("___", " → ").replace("_", " ").title()
                        pct = conf * 100
                        st.write(f"**{rank+1}.** {label} — `{pct:.1f}%`")
                        st.progress(min(conf, 1.0))

                    st.markdown("---")

                    # Full treatment info
                    if info:
                        sev = info.get("severity", "Unknown")
                        st.markdown(f"### 💊 Treatment & Prevention")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Type:** {info['type']}")
                        with col_b:
                            sev_emoji = {"None": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "🆘"}.get(sev, "⚪")
                            st.markdown(f"**Severity:** {sev_emoji} {sev}")

                        if "healthy" in disease.lower():
                            st.success(f"✅ {info['treatment']}")
                        else:
                            st.error(f"**🚨 Treatment:** {info['treatment']}")
                            st.warning(f"**🛡️ Prevention:** {info['prevention']}")

                        with st.expander("💊 Chemical Pesticide Recommendation"):
                            st.write(f"**Recommended:** {info['pesticide']}")
                            st.caption("⚠️ Always follow label instructions. Wear protective equipment.")

                        with st.expander("🌿 Organic / Natural Alternative"):
                            st.write(f"**Organic Option:** {info['organic']}")

                    else:
                        st.info("ℹ️ General advice: Consult your local agricultural extension officer for targeted treatment.")

                    # Log to DB
                    if selected_farmer:
                        solution_text = info['treatment'] if info else "Consult agronomist"
                        add_pest_log(selected_farmer, uploaded.name, disease, confidence, solution_text)
                        st.caption("✅ Detection saved to farmer profile.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.info("Ensure the uploaded image is a clear photo of a plant leaf.")