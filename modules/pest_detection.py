import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Path Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# ── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_disease_model():
    return load_model(MODEL_PATH, compile=False)

def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

# ── Disease Solutions ─────────────────────────────────────────────────────────
DISEASE_SOLUTIONS = {
    "Apple__Apple_scab": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Captan or Mancozeb fungicide every 7-10 days.",
        "organic": "Neem oil spray. Remove fallen leaves.",
        "cultural": "Prune for airflow. Use resistant varieties.",
        "prevention": "Rake and destroy fallen leaves in autumn."
    },
    "Apple__Black_rot": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Thiophanate-methyl or Captan fungicide.",
        "organic": "Copper-based fungicide spray.",
        "cultural": "Remove mummified fruits and dead wood.",
        "prevention": "Maintain tree vigor with proper fertilization."
    },
    "Apple__Cedar_apple_rust": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Myclobutanil or Propiconazole at bud break.",
        "organic": "Sulfur-based fungicide spray.",
        "cultural": "Remove nearby juniper/cedar trees if possible.",
        "prevention": "Plant resistant apple varieties."
    },
    "Apple__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Continue preventive neem oil spray monthly.",
        "cultural": "Maintain good pruning practices.",
        "prevention": "Regular monitoring for early detection."
    },
    "Corn__Cercospora_leaf_spot Gray_leaf_spot": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Azoxystrobin or Propiconazole fungicide.",
        "organic": "Copper hydroxide spray.",
        "cultural": "Crop rotation. Reduce plant density.",
        "prevention": "Use resistant hybrids. Avoid overhead irrigation."
    },
    "Corn__Common_rust": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Trifloxystrobin or Propiconazole at first sign.",
        "organic": "Sulfur dust application.",
        "cultural": "Plant resistant varieties. Early planting.",
        "prevention": "Monitor fields regularly during warm humid weather."
    },
    "Corn__Northern_Leaf_Blight": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Azoxystrobin + Propiconazole at tasseling.",
        "organic": "Biopesticide Bacillus subtilis spray.",
        "cultural": "Crop rotation with non-host crops.",
        "prevention": "Plant resistant hybrids."
    },
    "Corn__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "No treatment needed.",
        "cultural": "Maintain proper spacing and fertilization.",
        "prevention": "Continue scouting weekly."
    },
    "Grape__Black_rot": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Mancozeb or Myclobutanil before bloom.",
        "organic": "Copper sulfate spray (Bordeaux mixture).",
        "cultural": "Remove mummified berries. Improve canopy airflow.",
        "prevention": "Prune properly. Avoid wetting foliage."
    },
    "Grape__Esca_(Black_Measles)": {
        "type": "Fungal", "severity": "High",
        "chemical": "Protect pruning wounds with fungicide paste.",
        "organic": "Trichoderma-based biological treatment on wounds.",
        "cultural": "Remove and burn infected wood. Sterilize pruning tools.",
        "prevention": "Avoid large pruning cuts. Protect wounds immediately."
    },
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Mancozeb or Copper oxychloride.",
        "organic": "Neem oil + copper soap spray.",
        "cultural": "Improve ventilation. Remove infected leaves.",
        "prevention": "Avoid overhead watering."
    },
    "Grape__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Compost mulch. Neem oil preventive spray.",
        "cultural": "Remove old fruiting canes after harvest.",
        "prevention": "Monitor for cane blight and botrytis."
    },
    "Potato__Early_blight": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Mancozeb every 7 days.",
        "organic": "Copper-based fungicide or baking soda spray.",
        "cultural": "Remove lower infected leaves. Avoid overhead irrigation.",
        "prevention": "Use certified disease-free seed potatoes."
    },
    "Potato__Late_blight": {
        "type": "Fungal/Oomycete", "severity": "Very High",
        "chemical": "Apply Metalaxyl + Mancozeb immediately. Repeat every 5-7 days.",
        "organic": "Copper hydroxide spray. Remove and destroy infected plants.",
        "cultural": "Destroy volunteer plants. Avoid wet foliage.",
        "prevention": "Plant resistant varieties. Monitor during cool wet weather."
    },
    "Potato__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Compost application for soil health.",
        "cultural": "Hill soil around plants.",
        "prevention": "Rotate crops every 3 years."
    },
    "Rice__Bacterial_leaf_blight": {
        "type": "Bacterial", "severity": "High",
        "chemical": "Apply Copper oxychloride 50 WP @ 3g/liter.",
        "organic": "Pseudomonas fluorescens bioagent spray.",
        "cultural": "Drain fields. Use balanced nitrogen fertilizer.",
        "prevention": "Use resistant varieties. Treat seeds before sowing."
    },
    "Rice__Brown_spot": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Edifenphos or Tricyclazole fungicide.",
        "organic": "Neem seed kernel extract spray.",
        "cultural": "Balanced NPK fertilization. Avoid water stress.",
        "prevention": "Use resistant varieties and clean seeds."
    },
    "Rice__Leaf_smut": {
        "type": "Fungal", "severity": "Low",
        "chemical": "Seed treatment with Carbendazim 50 WP.",
        "organic": "Trichoderma viride seed treatment.",
        "cultural": "Deep summer plowing. Remove infected stubble.",
        "prevention": "Use certified seeds. Crop rotation."
    },
    "Rice__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "No treatment needed.",
        "cultural": "Maintain proper water management.",
        "prevention": "Regular field scouting."
    },
    "Tomato__Bacterial_spot": {
        "type": "Bacterial", "severity": "High",
        "chemical": "Apply Copper bactericide + Mancozeb.",
        "organic": "Copper soap spray. Remove infected leaves.",
        "cultural": "Use drip irrigation. Stake plants for airflow.",
        "prevention": "Use certified disease-free transplants."
    },
    "Tomato__Early_blight": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Mancozeb every 7-10 days.",
        "organic": "Copper fungicide spray. Mulch around plants.",
        "cultural": "Remove lower leaves touching soil.",
        "prevention": "Rotate tomatoes every 2-3 years."
    },
    "Tomato__Late_blight": {
        "type": "Fungal/Oomycete", "severity": "Very High",
        "chemical": "Apply Metalaxyl or Cymoxanil + Mancozeb immediately.",
        "organic": "Copper-based spray. Remove infected tissue.",
        "cultural": "Improve drainage. Reduce leaf wetness.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation."
    },
    "Tomato__Leaf_Mold": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Mancozeb.",
        "organic": "Baking soda + neem oil spray.",
        "cultural": "Reduce humidity in greenhouse. Improve ventilation.",
        "prevention": "Avoid wetting foliage when watering."
    },
    "Tomato__Septoria_leaf_spot": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Copper fungicide at first sign.",
        "organic": "Copper soap spray. Remove infected leaves.",
        "cultural": "Mulch to prevent soil splash. Remove lower leaves.",
        "prevention": "Crop rotation. Use disease-free transplants."
    },
    "Tomato__Spider_mites Two-spotted_spider_mite": {
        "type": "Pest (Mite)", "severity": "Medium",
        "chemical": "Apply Abamectin or Bifenazate miticide.",
        "organic": "Neem oil or insecticidal soap spray.",
        "cultural": "Avoid dusty conditions. Overhead water sprays help.",
        "prevention": "Monitor undersides of leaves regularly."
    },
    "Tomato__Target_Spot": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Azoxystrobin or Boscalid fungicide.",
        "organic": "Copper-based fungicide spray.",
        "cultural": "Remove infected leaves. Improve air circulation.",
        "prevention": "Avoid dense planting. Use mulch."
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "type": "Viral (Whitefly vector)", "severity": "Very High",
        "chemical": "Control whiteflies with Imidacloprid or Thiamethoxam.",
        "organic": "Yellow sticky traps. Neem oil spray for whiteflies.",
        "cultural": "Remove and destroy infected plants immediately.",
        "prevention": "Use virus-resistant varieties. Install insect-proof netting."
    },
    "Tomato__Tomato_mosaic_virus": {
        "type": "Viral", "severity": "High",
        "chemical": "No chemical cure. Disinfect tools with 10% bleach solution.",
        "organic": "Remove and destroy infected plants.",
        "cultural": "Wash hands before handling plants. Control aphids.",
        "prevention": "Use certified virus-free seeds. Resistant varieties."
    },
    "Tomato__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Compost tea spray for plant immunity.",
        "cultural": "Maintain proper staking and pruning.",
        "prevention": "Scout weekly for early pest/disease signs."
    },
    "Wheat__Brown_rust": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Propiconazole 25 EC @ 1ml/liter at flag leaf stage.",
        "organic": "Sulfur dust application.",
        "cultural": "Avoid late sowing. Balanced fertilization.",
        "prevention": "Use resistant varieties. Early season monitoring."
    },
    "Wheat__Yellow_rust": {
        "type": "Fungal", "severity": "Very High",
        "chemical": "Apply Tebuconazole or Propiconazole immediately.",
        "organic": "Sulfur-based fungicide spray.",
        "cultural": "Avoid excessive nitrogen. Early planting.",
        "prevention": "Plant resistant varieties. Use certified seeds."
    },
    "Wheat__healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "No treatment needed.",
        "cultural": "Balanced fertilization and proper irrigation.",
        "prevention": "Regular field monitoring."
    },
}

# ── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_disease(image: Image.Image):
    model = load_disease_model()
    idx_to_class = load_class_indices()
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    class_name = idx_to_class.get(predicted_index, "Unknown")
    solution = DISEASE_SOLUTIONS.get(class_name, {
        "type": "Unknown", "severity": "Unknown",
        "chemical": "Consult a local agricultural expert.",
        "organic": "Consult a local agricultural expert.",
        "cultural": "Monitor and isolate affected plants.",
        "prevention": "Regular crop scouting recommended."
    })
    return class_name, confidence, solution

# ── Streamlit Page ────────────────────────────────────────────────────────────
def pest_detection_page():
    st.title("🔍 Pest & Disease Detection")
    st.markdown("Upload a clear image of the affected plant leaf for AI-based diagnosis.")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            class_name, confidence, solution = predict_disease(image)

        display_name = class_name.replace("__", " → ").replace("_", " ")
        st.success(f"**Detected:** {display_name}")
        st.metric("Confidence", f"{confidence:.1f}%")

        severity_icon = {"None": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Very High": "🚨"}.get(solution["severity"], "⚪")
        st.markdown(f"### {severity_icon} Severity: **{solution['severity']}** | Type: **{solution['type']}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💊 Chemical Treatment")
            st.info(solution["chemical"])
            st.markdown("#### 🌿 Organic Treatment")
            st.success(solution["organic"])
        with col2:
            st.markdown("#### 🌾 Cultural Practices")
            st.warning(solution["cultural"])
            st.markdown("#### 🛡️ Prevention")
            st.error(solution["prevention"])