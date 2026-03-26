import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from database.db import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

@st.cache_resource
def load_disease_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        indices = json.load(f)
    return {v: k for k, v in indices.items()}

DISEASE_SOLUTIONS = {
    "Apple__Apple_scab":            {"type":"Fungal","severity":"Medium","chemical":"Apply Captan or Mancozeb every 7-10 days.","organic":"Neem oil spray. Remove fallen leaves.","cultural":"Prune for airflow.","prevention":"Rake fallen leaves in autumn."},
    "Apple__Black_rot":             {"type":"Fungal","severity":"High","chemical":"Apply Thiophanate-methyl or Captan.","organic":"Copper-based fungicide.","cultural":"Remove mummified fruits.","prevention":"Maintain tree vigor."},
    "Apple__Cedar_apple_rust":      {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil at bud break.","organic":"Sulfur-based spray.","cultural":"Remove nearby cedar trees.","prevention":"Plant resistant varieties."},
    "Apple__healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive neem oil monthly.","cultural":"Maintain pruning.","prevention":"Regular monitoring."},
    "Corn__Cercospora_leaf_spot Gray_leaf_spot": {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin.","organic":"Copper hydroxide spray.","cultural":"Crop rotation.","prevention":"Use resistant hybrids."},
    "Corn__Common_rust":            {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin.","organic":"Sulfur dust.","cultural":"Plant resistant varieties.","prevention":"Monitor during humid weather."},
    "Corn__Northern_Leaf_Blight":   {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin + Propiconazole.","organic":"Bacillus subtilis spray.","cultural":"Crop rotation.","prevention":"Plant resistant hybrids."},
    "Corn__healthy":                {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Maintain proper spacing.","prevention":"Weekly scouting."},
    "Grape__Black_rot":             {"type":"Fungal","severity":"High","chemical":"Apply Mancozeb before bloom.","organic":"Bordeaux mixture.","cultural":"Remove mummified berries.","prevention":"Prune properly."},
    "Grape__Esca_(Black_Measles)":  {"type":"Fungal","severity":"High","chemical":"Protect pruning wounds with fungicide paste.","organic":"Trichoderma on wounds.","cultural":"Remove infected wood.","prevention":"Avoid large cuts."},
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)": {"type":"Fungal","severity":"Medium","chemical":"Apply Mancozeb.","organic":"Neem oil + copper soap.","cultural":"Remove infected leaves.","prevention":"Avoid overhead watering."},
    "Grape__healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Neem oil preventive spray.","cultural":"Remove old canes.","prevention":"Monitor for blight."},
    "Potato__Early_blight":         {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil every 7 days.","organic":"Copper-based spray.","cultural":"Remove infected leaves.","prevention":"Use certified seeds."},
    "Potato__Late_blight":          {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl + Mancozeb immediately.","organic":"Copper hydroxide spray.","cultural":"Destroy infected plants.","prevention":"Plant resistant varieties."},
    "Potato__healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost application.","cultural":"Hill soil around plants.","prevention":"Rotate crops every 3 years."},
    "Rice__Bacterial_leaf_blight":  {"type":"Bacterial","severity":"High","chemical":"Copper oxychloride 50 WP @ 3g/liter.","organic":"Pseudomonas fluorescens spray.","cultural":"Drain fields.","prevention":"Use resistant varieties."},
    "Rice__Brown_spot":             {"type":"Fungal","severity":"Medium","chemical":"Apply Edifenphos.","organic":"Neem seed kernel extract.","cultural":"Balanced NPK fertilization.","prevention":"Use resistant varieties."},
    "Rice__Leaf_smut":              {"type":"Fungal","severity":"Low","chemical":"Carbendazim 50 WP seed treatment.","organic":"Trichoderma viride.","cultural":"Deep summer plowing.","prevention":"Use certified seeds."},
    "Rice__healthy":                {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Proper water management.","prevention":"Regular field scouting."},
    "Tomato__Bacterial_spot":       {"type":"Bacterial","severity":"High","chemical":"Copper bactericide + Mancozeb.","organic":"Copper soap spray.","cultural":"Use drip irrigation.","prevention":"Disease-free transplants."},
    "Tomato__Early_blight":         {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil every 7-10 days.","organic":"Copper fungicide + mulch.","cultural":"Remove lower leaves.","prevention":"Rotate every 2-3 years."},
    "Tomato__Late_blight":          {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl immediately.","organic":"Copper-based spray.","cultural":"Improve drainage.","prevention":"Avoid overhead irrigation."},
    "Tomato__Leaf_Mold":            {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil.","organic":"Baking soda + neem oil.","cultural":"Reduce humidity.","prevention":"Avoid wetting foliage."},
    "Tomato__Septoria_leaf_spot":   {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil at first sign.","organic":"Copper soap spray.","cultural":"Mulch to prevent soil splash.","prevention":"Crop rotation."},
    "Tomato__Spider_mites Two-spotted_spider_mite": {"type":"Pest (Mite)","severity":"Medium","chemical":"Apply Abamectin.","organic":"Neem oil or insecticidal soap.","cultural":"Avoid dusty conditions.","prevention":"Monitor leaf undersides."},
    "Tomato__Target_Spot":          {"type":"Fungal","severity":"Medium","chemical":"Apply Azoxystrobin.","organic":"Copper-based spray.","cultural":"Remove infected leaves.","prevention":"Avoid dense planting."},
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {"type":"Viral","severity":"Very High","chemical":"Control whiteflies with Imidacloprid.","organic":"Yellow sticky traps + neem oil.","cultural":"Destroy infected plants immediately.","prevention":"Use virus-resistant varieties."},
    "Tomato__Tomato_mosaic_virus":  {"type":"Viral","severity":"High","chemical":"No cure. Disinfect tools with 10% bleach.","organic":"Remove infected plants.","cultural":"Wash hands. Control aphids.","prevention":"Certified virus-free seeds."},
    "Tomato__healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost tea spray.","cultural":"Maintain staking and pruning.","prevention":"Scout weekly."},
    "Wheat__Brown_rust":            {"type":"Fungal","severity":"High","chemical":"Propiconazole 25 EC @ 1ml/liter.","organic":"Sulfur dust.","cultural":"Avoid late sowing.","prevention":"Use resistant varieties."},
    "Wheat__Yellow_rust":           {"type":"Fungal","severity":"Very High","chemical":"Apply Tebuconazole immediately.","organic":"Sulfur-based spray.","cultural":"Avoid excess nitrogen.","prevention":"Plant resistant varieties."},
    "Wheat__healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Balanced fertilization.","prevention":"Regular monitoring."},
    "Pepper__bell__Bacterial_spot": {"type":"Bacterial","severity":"High","chemical":"Copper bactericide spray.","organic":"Copper soap + neem oil.","cultural":"Drip irrigation. Remove infected leaves.","prevention":"Use disease-free seeds."},
    "Pepper__bell__healthy":        {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive neem oil spray.","cultural":"Proper staking.","prevention":"Monitor weekly."},
    "Strawberry__Leaf_scorch":      {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil or Captan.","organic":"Copper-based spray.","cultural":"Remove infected leaves.","prevention":"Avoid overhead irrigation."},
    "Strawberry__healthy":          {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost mulch.","cultural":"Remove old leaves after harvest.","prevention":"Monitor for grey mold."},
    "Squash__Powdery_mildew":       {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin at first sign.","organic":"Baking soda 5g/liter + neem oil weekly.","cultural":"Improve air circulation.","prevention":"Plant resistant varieties."},
    "Soybean__healthy":             {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Rhizobium inoculant at planting.","cultural":"Maintain 45cm row spacing.","prevention":"Monitor for sudden death syndrome."},
    "Peach__Bacterial_spot":        {"type":"Bacterial","severity":"High","chemical":"Apply Oxytetracycline or copper fungicide.","organic":"Copper hydroxide spray.","cultural":"Prune to improve airflow.","prevention":"Plant resistant varieties."},
    "Peach__healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive copper spray in spring.","cultural":"Thin fruits for airflow.","prevention":"Monitor for brown rot."},
    "Cherry__(including_sour)__Powdery_mildew": {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil.","organic":"Sulfur spray.","cultural":"Prune infected shoots.","prevention":"Avoid excessive nitrogen."},
    "Cherry__(including_sour)__healthy": {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Neem oil preventive spray.","cultural":"Maintain tree shape.","prevention":"Monitor for cherry leaf spot."},
}

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_disease(image: Image.Image):
    model = load_disease_model()
    idx_to_class = load_class_indices()
    predictions = model.predict(preprocess_image(image))
    idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    class_name = idx_to_class.get(idx, "Unknown")
    solution = DISEASE_SOLUTIONS.get(class_name, {
        "type": "Unknown", "severity": "Unknown",
        "chemical": "Consult a local agricultural expert.",
        "organic": "Consult a local agricultural expert.",
        "cultural": "Monitor and isolate affected plants.",
        "prevention": "Regular crop scouting recommended."
    })
    return class_name, confidence, solution

def pest_detection_page():
    st.title("🔍 Pest & Disease Detection")

    conn = get_connection()
    farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
    conn.close()

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()))
    uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Disease", type="primary"):
            with st.spinner("Analyzing image with AI..."):
                class_name, confidence, solution = predict_disease(image)

            display_name = class_name.replace("__", " → ").replace("_", " ")
            st.success(f"**Detected:** {display_name}")
            st.metric("Confidence", f"{confidence:.1f}%")

            sev_icon = {"None":"🟢","Low":"🟡","Medium":"🟠","High":"🔴","Very High":"🚨"}.get(solution["severity"],"⚪")
            st.markdown(f"### {sev_icon} Severity: **{solution['severity']}** | Type: **{solution['type']}**")

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

            # Save to DB
            fid = farmer_options[farmer_key]
            crop_part = class_name.split("__")[0] if "__" in class_name else class_name
            conn = get_connection()
            conn.execute("""
                INSERT INTO pest_detections (farmer_id, crop_name, disease_detected, confidence, treatment)
                VALUES (?,?,?,?,?)
            """, (fid, crop_part, display_name, confidence, solution["chemical"]))
            conn.commit()
            conn.close()
            st.info("📝 Detection saved to database.")