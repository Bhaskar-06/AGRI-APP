import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from database.db import get_connection

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH        = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_INDICES_PATH= os.path.join(BASE_DIR, "models", "class_indices.json")

@st.cache_resource
def load_disease_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        indices = json.load(f)
    return {v: k for k, v in indices.items()}

DISEASE_SOLUTIONS = {
    "Apple___Apple_scab":            {"type":"Fungal","severity":"Medium","chemical":"Apply Captan or Mancozeb every 7-10 days.","organic":"Neem oil spray. Remove fallen leaves.","cultural":"Prune for airflow. Use resistant varieties.","prevention":"Rake and destroy fallen leaves in autumn."},
    "Apple___Black_rot":             {"type":"Fungal","severity":"High","chemical":"Apply Thiophanate-methyl or Captan fungicide.","organic":"Copper-based fungicide spray.","cultural":"Remove mummified fruits and dead wood.","prevention":"Maintain tree vigor with proper fertilization."},
    "Apple___Cedar_apple_rust":      {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil or Propiconazole at bud break.","organic":"Sulfur-based fungicide spray.","cultural":"Remove nearby juniper/cedar trees if possible.","prevention":"Plant resistant apple varieties."},
    "Apple___healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive neem oil spray monthly.","cultural":"Maintain good pruning practices.","prevention":"Regular monitoring."},
    "Blueberry___healthy":           {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Mulch with pine bark.","cultural":"Maintain acidic soil pH 4.5-5.5.","prevention":"Scout regularly for mummy berry."},
    "Cherry_(including_sour)___Powdery_mildew": {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil.","organic":"Sulfur spray.","cultural":"Prune infected shoots.","prevention":"Avoid excessive nitrogen."},
    "Cherry_(including_sour)___healthy": {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Neem oil preventive spray.","cultural":"Maintain tree shape.","prevention":"Monitor for cherry leaf spot."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin or Propiconazole.","organic":"Copper hydroxide spray.","cultural":"Crop rotation. Reduce plant density.","prevention":"Use resistant hybrids."},
    "Corn_(maize)___Common_rust_":   {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin at first sign.","organic":"Sulfur dust application.","cultural":"Plant resistant varieties. Early planting.","prevention":"Monitor during warm humid weather."},
    "Corn_(maize)___Northern_Leaf_Blight": {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin + Propiconazole at tasseling.","organic":"Bacillus subtilis spray.","cultural":"Crop rotation with non-host crops.","prevention":"Plant resistant hybrids."},
    "Corn_(maize)___healthy":        {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Maintain proper spacing.","prevention":"Weekly scouting."},
    "Grape___Black_rot":             {"type":"Fungal","severity":"High","chemical":"Apply Mancozeb or Myclobutanil before bloom.","organic":"Copper sulfate spray (Bordeaux mixture).","cultural":"Remove mummified berries. Improve canopy airflow.","prevention":"Prune properly. Avoid wetting foliage."},
    "Grape___Esca_(Black_Measles)":  {"type":"Fungal","severity":"High","chemical":"Protect pruning wounds with fungicide paste.","organic":"Trichoderma-based biological treatment on wounds.","cultural":"Remove and burn infected wood. Sterilize pruning tools.","prevention":"Avoid large pruning cuts."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"type":"Fungal","severity":"Medium","chemical":"Apply Mancozeb or Copper oxychloride.","organic":"Neem oil + copper soap spray.","cultural":"Improve ventilation. Remove infected leaves.","prevention":"Avoid overhead watering."},
    "Grape___healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost mulch. Neem oil preventive spray.","cultural":"Remove old fruiting canes after harvest.","prevention":"Monitor for cane blight and botrytis."},
    "Orange___Haunglongbing_(Citrus_greening)": {"type":"Bacterial","severity":"Very High","chemical":"Control psyllid with Imidacloprid.","organic":"Neem oil spray for psyllid control.","cultural":"Remove and destroy infected trees immediately.","prevention":"Use certified disease-free nursery stock."},
    "Peach___Bacterial_spot":        {"type":"Bacterial","severity":"High","chemical":"Apply Oxytetracycline or copper fungicide.","organic":"Copper hydroxide spray.","cultural":"Prune to improve airflow.","prevention":"Plant resistant varieties."},
    "Peach___healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive copper spray in spring.","cultural":"Thin fruits for airflow.","prevention":"Monitor for brown rot."},
    "Pepper,_bell___Bacterial_spot": {"type":"Bacterial","severity":"High","chemical":"Copper bactericide spray.","organic":"Copper soap + neem oil.","cultural":"Drip irrigation. Remove infected leaves.","prevention":"Use disease-free seeds."},
    "Pepper,_bell___healthy":        {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive neem oil spray.","cultural":"Proper staking.","prevention":"Monitor weekly."},
    "Potato___Early_blight":         {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil or Mancozeb every 7 days.","organic":"Copper-based fungicide or baking soda spray.","cultural":"Remove lower infected leaves.","prevention":"Use certified disease-free seed potatoes."},
    "Potato___Late_blight":          {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl + Mancozeb immediately.","organic":"Copper hydroxide spray.","cultural":"Destroy volunteer plants.","prevention":"Plant resistant varieties."},
    "Potato___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost application.","cultural":"Hill soil around plants.","prevention":"Rotate crops every 3 years."},
    "Raspberry___healthy":           {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Neem oil spray.","cultural":"Prune old canes after harvest.","prevention":"Monitor for cane diseases."},
    "Soybean___healthy":             {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Rhizobium inoculant at planting.","cultural":"Maintain 45cm row spacing.","prevention":"Monitor for sudden death syndrome."},
    "Squash___Powdery_mildew":       {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin at first sign.","organic":"Baking soda 5g/liter + neem oil weekly.","cultural":"Improve air circulation.","prevention":"Plant resistant varieties."},
    "Strawberry___Leaf_scorch":      {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil or Captan.","organic":"Copper-based spray.","cultural":"Remove infected leaves.","prevention":"Avoid overhead irrigation."},
    "Strawberry___healthy":          {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost mulch.","cultural":"Remove old leaves after harvest.","prevention":"Monitor for grey mold."},
    "Tomato___Bacterial_spot":       {"type":"Bacterial","severity":"High","chemical":"Apply Copper bactericide + Mancozeb.","organic":"Copper soap spray.","cultural":"Use drip irrigation.","prevention":"Use certified disease-free transplants."},
    "Tomato___Early_blight":         {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil or Mancozeb every 7-10 days.","organic":"Copper fungicide spray. Mulch around plants.","cultural":"Remove lower leaves touching soil.","prevention":"Rotate tomatoes every 2-3 years."},
    "Tomato___Late_blight":          {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl or Cymoxanil + Mancozeb immediately.","organic":"Copper-based spray.","cultural":"Improve drainage.","prevention":"Avoid overhead irrigation."},
    "Tomato___Leaf_Mold":            {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil or Mancozeb.","organic":"Baking soda + neem oil.","cultural":"Reduce humidity in greenhouse.","prevention":"Avoid wetting foliage."},
    "Tomato___Septoria_leaf_spot":   {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil at first sign.","organic":"Copper soap spray.","cultural":"Mulch to prevent soil splash.","prevention":"Crop rotation."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"type":"Pest (Mite)","severity":"Medium","chemical":"Apply Abamectin or Bifenazate.","organic":"Neem oil or insecticidal soap.","cultural":"Avoid dusty conditions.","prevention":"Monitor leaf undersides."},
    "Tomato___Target_Spot":          {"type":"Fungal","severity":"Medium","chemical":"Apply Azoxystrobin or Boscalid.","organic":"Copper-based spray.","cultural":"Remove infected leaves.","prevention":"Avoid dense planting."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"type":"Viral","severity":"Very High","chemical":"Control whiteflies with Imidacloprid.","organic":"Yellow sticky traps + neem oil.","cultural":"Remove infected plants immediately.","prevention":"Use virus-resistant varieties."},
    "Tomato___Tomato_mosaic_virus":  {"type":"Viral","severity":"High","chemical":"No cure. Disinfect tools with 10% bleach.","organic":"Remove infected plants.","cultural":"Wash hands. Control aphids.","prevention":"Certified virus-free seeds."},
    "Tomato___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost tea spray.","cultural":"Maintain proper staking.","prevention":"Scout weekly."},
}

def normalize_key(raw_key: str) -> str:
    """Try multiple key format variations to find a match in DISEASE_SOLUTIONS."""
    candidates = [
        raw_key,
        raw_key.replace("__", "___"),
        raw_key.replace("___", "__"),
        raw_key.replace(",_", "___").replace("__", "___"),
    ]
    for candidate in candidates:
        if candidate in DISEASE_SOLUTIONS:
            return candidate
    # Fuzzy: try lowercase match
    lower_map = {k.lower(): k for k in DISEASE_SOLUTIONS}
    if raw_key.lower() in lower_map:
        return lower_map[raw_key.lower()]
    return None

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_disease(image: Image.Image):
    model        = load_disease_model()
    idx_to_class = load_class_indices()
    preds        = model.predict(preprocess_image(image))
    idx          = int(np.argmax(preds[0]))
    confidence   = float(np.max(preds[0])) * 100
    raw_name     = idx_to_class.get(idx, "Unknown")

    matched_key  = normalize_key(raw_name)
    solution     = DISEASE_SOLUTIONS.get(matched_key, {
        "type": "Unknown", "severity": "Unknown",
        "chemical": "Consult a local agricultural expert.",
        "organic":  "Consult a local agricultural expert.",
        "cultural": "Monitor and isolate affected plants.",
        "prevention": "Regular crop scouting recommended."
    })
    return raw_name, confidence, solution

def pest_detection_page():
    st.title("🔍 Pest & Disease Detection")

    try:
        conn    = get_connection()
        farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    farmer_key    = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="pd_farmer")
    uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg","jpeg","png"], key="pd_upload")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Disease", type="primary", key="pd_detect"):
            with st.spinner("Analyzing image with AI model..."):
                raw_name, confidence, solution = predict_disease(image)

            display_name = raw_name.replace("___", " → ").replace("__", " → ").replace("_", " ")
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
            try:
                fid       = farmer_options[farmer_key]
                crop_part = raw_name.split("___")[0].split("__")[0] if "___" in raw_name or "__" in raw_name else raw_name
                conn      = get_connection()
                conn.execute("""
                    INSERT INTO pest_detections (farmer_id, crop_name, disease_detected, confidence, treatment)
                    VALUES (?,?,?,?,?)
                """, (fid, crop_part, display_name, confidence, solution["chemical"]))
                conn.commit()
                conn.close()
                st.info("📝 Detection saved to database.")
            except Exception as e:
                st.error(f"Save error: {e}")