import streamlit as st
import numpy as np
from PIL import Image
import json, os, requests
import tensorflow as tf
from database.db import add_pest_log, get_all_farmers

MODEL_PATH = "models/plant_disease_model.h5"
INDEX_PATH = "models/class_indices.json"
MODEL_URL  = "https://huggingface.co/spaces/etahamad/plant-disease-detection/resolve/main/model.h5"

CLASS_INDICES = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry___Powdery_mildew",
    "6": "Cherry___healthy",
    "7": "Corn___Cercospora_leaf_spot",
    "8": "Corn___Common_rust",
    "9": "Corn___Northern_Leaf_Blight",
    "10": "Corn___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca",
    "13": "Grape___Leaf_blight",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper___Bacterial_spot",
    "19": "Pepper___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Yellow_Leaf_Curl_Virus",
    "36": "Tomato___mosaic_virus",
    "37": "Tomato___healthy"
}

# ─── Full Solutions for ALL 38 classes ───────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "type": "Fungal",
        "severity": "High",
        "chemical": "Apply Captan (2g/L) or Mancozeb (2.5g/L) every 7-10 days during wet weather.",
        "organic": "Spray neem oil (5ml/L) or baking soda solution (5g/L) on affected leaves.",
        "cultural": "Rake and destroy fallen leaves. Prune crowded branches for airflow. Avoid overhead irrigation.",
        "prevention": "Plant scab-resistant apple varieties. Apply dormant oil spray before bud break."
    },
    "Apple___Black_rot": {
        "type": "Fungal",
        "severity": "High",
        "chemical": "Apply Thiophanate-methyl or Captan fungicide at petal fall stage.",
        "organic": "Copper-based fungicide (Bordeaux mixture). Remove mummified fruits immediately.",
        "cultural": "Prune dead/infected wood. Sanitize pruning tools with 70% alcohol between cuts.",
        "prevention": "Remove all infected fruit from tree and ground. Avoid wounding fruit during harvest."
    },
    "Apple___Cedar_apple_rust": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Myclobutanil or Propiconazole fungicide from pink bud through early summer.",
        "organic": "Sulfur-based fungicide spray every 7 days during infection period.",
        "cultural": "Remove nearby cedar/juniper trees if possible. Prune galls from cedar trees in winter.",
        "prevention": "Plant rust-resistant apple varieties like Liberty or Freedom."
    },
    "Apple___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Continue regular neem oil preventive sprays monthly.",
        "cultural": "Maintain good orchard hygiene. Regular pruning for air circulation.",
        "prevention": "Monitor regularly. Apply balanced NPK fertilizer for strong immunity."
    },
    "Blueberry___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost mulch around base to retain moisture.",
        "cultural": "Maintain soil pH between 4.5-5.5 for optimal blueberry health.",
        "prevention": "Regular monitoring. Ensure adequate irrigation during fruiting."
    },
    "Cherry___Powdery_mildew": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Trifloxystrobin or Myclobutanil fungicide. Repeat every 10-14 days.",
        "organic": "Potassium bicarbonate (10g/L) or diluted milk spray (1:9 ratio) weekly.",
        "cultural": "Improve air circulation by thinning canopy. Avoid excess nitrogen fertilizer.",
        "prevention": "Plant resistant cherry varieties. Avoid overhead watering."
    },
    "Cherry___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive neem oil spray every 2-3 weeks.",
        "cultural": "Regular pruning. Balanced fertilization.",
        "prevention": "Monitor for early signs of powdery mildew especially in humid weather."
    },
    "Corn___Cercospora_leaf_spot": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Azoxystrobin + Propiconazole (Quilt Xcel) at VT/R1 stage.",
        "organic": "Copper hydroxide spray. Trichoderma-based biocontrol agents.",
        "cultural": "Crop rotation with non-host crops. Bury infected crop residues.",
        "prevention": "Use resistant hybrids. Avoid continuous corn cropping in same field."
    },
    "Corn___Common_rust": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Propiconazole or Trifloxystrobin fungicide at early rust appearance.",
        "organic": "Sulfur dust application. Neem oil spray (3ml/L) weekly.",
        "cultural": "Plant early-maturing varieties to escape peak infection periods.",
        "prevention": "Use rust-resistant corn hybrids. Scout fields regularly from V6 stage."
    },
    "Corn___Northern_Leaf_Blight": {
        "type": "Fungal",
        "severity": "High",
        "chemical": "Apply Azoxystrobin or Pyraclostrobin fungicide at first sign of lesions.",
        "organic": "Bacillus subtilis-based products (Serenade). Copper sulfate spray.",
        "cultural": "Rotate crops. Till infected residues. Plant after soil reaches 15°C.",
        "prevention": "Use NLBR-resistant hybrids. Avoid overhead irrigation."
    },
    "Corn___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost tea as foliar spray for immunity boost.",
        "cultural": "Maintain proper plant spacing (60-75cm row spacing).",
        "prevention": "Monitor weekly. Scout for aphids and armyworms."
    },
    "Grape___Black_rot": {
        "type": "Fungal",
        "severity": "High",
        "chemical": "Apply Myclobutanil (Rally) or Mancozeb every 7-14 days from bud break.",
        "organic": "Bordeaux mixture (1%) at bud break and bloom stages.",
        "cultural": "Remove all mummified berries from vine and ground. Prune for airflow.",
        "prevention": "Train vines to improve air circulation. Avoid leaf wetness."
    },
    "Grape___Esca": {
        "type": "Fungal (Wood disease)",
        "severity": "Very High",
        "chemical": "No curative fungicide available. Sodium arsenite (where legal) used historically.",
        "organic": "Trichoderma harzianum wound protectant on pruning cuts.",
        "cultural": "Remove and destroy infected wood. Disinfect pruning tools. Avoid large pruning wounds.",
        "prevention": "Protect pruning wounds immediately with wound sealant. Prune during dry weather."
    },
    "Grape___Leaf_blight": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Iprodione or Fludioxonil fungicide. Mancozeb as protectant.",
        "organic": "Copper oxychloride spray (3g/L). Increase spray frequency in humid weather.",
        "cultural": "Remove infected leaves promptly. Improve canopy ventilation.",
        "prevention": "Avoid water stress. Maintain balanced nutrition (avoid excess N)."
    },
    "Grape___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive Bordeaux mixture spray before monsoon.",
        "cultural": "Regular canopy management. Proper trellis training.",
        "prevention": "Monitor for downy/powdery mildew weekly."
    },
    "Orange___Haunglongbing": {
        "type": "Bacterial (Citrus Greening)",
        "severity": "CRITICAL - No Cure",
        "chemical": "No cure exists. Remove and destroy infected trees immediately.",
        "organic": "Neem oil spray to control Asian Citrus Psyllid (vector insect).",
        "cultural": "Quarantine infected trees. Use certified disease-free nursery plants only.",
        "prevention": "Control psyllid population with Imidacloprid. Plant in psyllid-free areas."
    },
    "Peach___Bacterial_spot": {
        "type": "Bacterial",
        "severity": "High",
        "chemical": "Apply copper hydroxide or oxytetracycline (Mycoshield) at bud swell. Repeat weekly.",
        "organic": "Copper sulfate spray (Bordeaux mixture) from dormant season.",
        "cultural": "Avoid overhead irrigation. Remove infected twigs during dormancy.",
        "prevention": "Plant resistant peach varieties. Windbreaks reduce spread."
    },
    "Peach___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Dormant copper spray as preventive measure.",
        "cultural": "Annual pruning for canopy airflow. Thin fruit for better size.",
        "prevention": "Monitor for peach leaf curl and brown rot."
    },
    "Pepper___Bacterial_spot": {
        "type": "Bacterial",
        "severity": "High",
        "chemical": "Apply copper hydroxide (Kocide) + Mancozeb weekly. Start at transplanting.",
        "organic": "Copper-based Bordeaux mixture. Biocontrol with Bacillus amyloliquefaciens.",
        "cultural": "Avoid overhead irrigation. Rotate away from solanaceous crops for 2 years.",
        "prevention": "Use certified disease-free seed. Hot water seed treatment (50°C for 25 min)."
    },
    "Pepper___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Foliar spray with compost tea for nutrient boost.",
        "cultural": "Stake plants. Ensure good drainage. Proper spacing (45cm).",
        "prevention": "Monitor for aphids and thrips which spread viruses."
    },
    "Potato___Early_blight": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Chlorothalonil (Bravo) or Mancozeb every 7-10 days from appearance.",
        "organic": "Neem oil (5ml/L) + copper soap spray. Bacillus subtilis products.",
        "cultural": "Crop rotation (3-year cycle). Remove infected leaves. Avoid excess nitrogen.",
        "prevention": "Plant certified disease-free seed potatoes. Hill soil around stems."
    },
    "Potato___Late_blight": {
        "type": "Fungal (Oomycete)",
        "severity": "CRITICAL",
        "chemical": "Apply Metalaxyl + Mancozeb (Ridomil Gold) immediately. Repeat every 5-7 days.",
        "organic": "Copper hydroxide spray (Bordeaux mixture 1%). Limited effectiveness.",
        "cultural": "Destroy all infected plant material by burning. Do not compost. Improve drainage.",
        "prevention": "Plant resistant varieties (Sarpo Mira). Avoid overhead irrigation. Monitor daily."
    },
    "Potato___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive copper spray before monsoon season.",
        "cultural": "Hill potatoes regularly. Ensure proper drainage.",
        "prevention": "Scout weekly. Watch for late blight during cool/wet weather."
    },
    "Raspberry___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost mulch. Neem oil preventive spray.",
        "cultural": "Remove old fruiting canes after harvest.",
        "prevention": "Monitor for cane blight and botrytis fruit rot."
    },
    "Soybean___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Rhizobium inoculant at planting for nitrogen fixation.",
        "cultural": "Maintain 45cm row spacing. Scout for soybean rust.",
        "prevention": "Monitor for sudden death syndrome and white mold."
    },
    "Squash___Powdery_mildew": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Trifloxystrobin or Tebuconazole fungicide at first sign.",
        "organic": "Baking soda solution (5g/L + 2ml dish soap). Neem oil weekly.",
        "cultural": "Improve air circulation. Remove infected leaves. Avoid evening watering.",
        "prevention": "Plant resistant varieties. Maintain dry foliage. Space plants adequately."
    },
    "Strawberry___Leaf_scorch": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Captan or Myclobutanil fungicide. Start in early spring.",
        "organic": "Copper hydroxide spray. Remove infected leaves immediately.",
        "cultural": "Avoid overhead irrigation. Renovate beds after harvest. Remove old leaves.",
        "prevention": "Plant certified disease-free runners. Avoid poorly drained soils."
    },
    "Strawberry___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive neem oil spray. Compost mulch to prevent soil splash.",
        "cultural": "Renew beds every 3-4 years. Remove runners to reduce overcrowding.",
        "prevention": "Monitor for gray mold (Botrytis) during wet seasons."
    },
    "Tomato___Bacterial_spot": {
        "type": "Bacterial",
        "severity": "High",
        "chemical": "Apply copper hydroxide (Kocide 3000) + Mancozeb weekly.",
        "organic": "Bordeaux mixture (1%). Bacillus subtilis biocontrol spray.",
        "cultural": "Avoid overhead irrigation. Stake and prune for airflow. Rotate crops 2 years.",
        "prevention": "Use disease-free transplants. Hot water seed treatment (50°C, 25 min)."
    },
    "Tomato___Early_blight": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Mancozeb every 7-10 days from first lesion.",
        "organic": "Copper fungicide spray. Neem oil (5ml/L) weekly.",
        "cultural": "Remove lower infected leaves. Mulch to prevent soil splash. Crop rotation.",
        "prevention": "Avoid overhead watering. Adequate plant spacing (60cm). Balanced NPK."
    },
    "Tomato___Late_blight": {
        "type": "Fungal (Oomycete)",
        "severity": "CRITICAL",
        "chemical": "Apply Metalaxyl + Mancozeb (Ridomil) or Cymoxanil immediately. Every 5 days.",
        "organic": "Copper sulfate (Bordeaux 1%). Limited effectiveness at late stage.",
        "cultural": "Destroy all infected plants immediately. Do not compost. Improve drainage.",
        "prevention": "Avoid overhead irrigation. Plant resistant varieties. Monitor during rain."
    },
    "Tomato___Leaf_Mold": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Mancozeb. Rotate with Iprodione every 14 days.",
        "organic": "Neem oil + potassium bicarbonate spray. Increase ventilation in greenhouse.",
        "cultural": "Reduce humidity below 85%. Remove infected leaves. Prune for airflow.",
        "prevention": "Use resistant tomato varieties. Avoid leaf wetness. Space plants 60-90cm."
    },
    "Tomato___Septoria_leaf_spot": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Mancozeb or Chlorothalonil at first sign. Repeat every 7-10 days.",
        "organic": "Copper-based fungicide. Neem oil spray.",
        "cultural": "Remove infected lower leaves. Stake plants. Avoid wetting foliage.",
        "prevention": "Crop rotation (3 years). Mulch soil surface. Plant resistant varieties."
    },
    "Tomato___Spider_mites": {
        "type": "Pest (Arachnid)",
        "severity": "Medium",
        "chemical": "Apply Abamectin or Bifenazate miticide. Rotate chemicals to prevent resistance.",
        "organic": "Neem oil (5ml/L) spray. Insecticidal soap spray. Release predatory mites.",
        "cultural": "Maintain adequate soil moisture. Remove heavily infested leaves.",
        "prevention": "Avoid water stress. Monitor undersides of leaves weekly for webbing."
    },
    "Tomato___Target_Spot": {
        "type": "Fungal",
        "severity": "Medium",
        "chemical": "Apply Chlorothalonil or Azoxystrobin fungicide every 7-14 days.",
        "organic": "Copper hydroxide spray. Trichoderma-based biocontrol.",
        "cultural": "Remove infected leaves. Improve air circulation. Avoid overhead irrigation.",
        "prevention": "Use resistant varieties. Stake plants. Mulch around base."
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "type": "Viral (Whitefly transmitted)",
        "severity": "Very High - No Cure",
        "chemical": "No cure for virus. Control whitefly vector with Imidacloprid or Thiamethoxam.",
        "organic": "Yellow sticky traps for whiteflies. Neem oil spray to deter vectors.",
        "cultural": "Remove and destroy infected plants immediately. Use reflective mulches.",
        "prevention": "Use virus-resistant tomato varieties. Install insect-proof nets in nursery."
    },
    "Tomato___mosaic_virus": {
        "type": "Viral (Contact transmitted)",
        "severity": "High - No Cure",
        "chemical": "No chemical cure. Disinfect tools with 10% bleach solution between plants.",
        "organic": "Remove infected plants. Wash hands thoroughly after touching infected plants.",
        "cultural": "Avoid handling infected plants. Control aphid and thrips vectors.",
        "prevention": "Use TMV-resistant varieties. Seed treatment with trisodium phosphate."
    },
    "Tomato___healthy": {
        "type": "None",
        "severity": "None",
        "chemical": "No treatment needed. Apply preventive copper spray before monsoon.",
        "organic": "Foliar spray with compost tea or fish emulsion for nutrients.",
        "cultural": "Stake plants. Prune suckers. Mulch soil to retain moisture.",
        "prevention": "Scout weekly. Monitor for early blight and late blight especially in wet weather."
    }
}

def ensure_model_exists():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading disease detection model for first time use...")
        with st.spinner("Downloading model (~3MB)..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Model downloaded!")
    if not os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "w") as f:
            json.dump(CLASS_INDICES, f)

@st.cache_resource
def load_model():
    ensure_model_exists()
    return tf.keras.models.load_model(MODEL_PATH)

def predict_disease(image):
    model = load_model()
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_idx = str(int(np.argmax(predictions)))
    disease = CLASS_INDICES.get(class_idx, "Unknown Disease")
    return disease, confidence, predictions[0]

def show_solution_card(disease, confidence):
    info = DISEASE_INFO.get(disease, None)
    clean_name = disease.replace("___", " → ").replace("_", " ")

    # Confidence warning
    if confidence < 0.50:
        st.warning(f"Low confidence ({confidence*100:.1f}%). Try a clearer, well-lit image of the leaf.")
    elif confidence < 0.75:
        st.info(f"Moderate confidence ({confidence*100:.1f}%). Result is likely correct but verify visually.")
    else:
        st.success(f"High confidence ({confidence*100:.1f}%).")

    if info is None:
        st.error("No solution data found for this disease.")
        return

    # Severity badge
    severity_colors = {
        "None": "green", "Medium": "orange",
        "High": "red", "Very High": "red",
        "CRITICAL": "red", "CRITICAL - No Cure": "red"
    }
    sev = info["severity"]
    col1, col2 = st.columns(2)
    col1.metric("Disease Type", info["type"])
    col2.metric("Severity", sev)

    if "healthy" in disease.lower():
        st.success("Your plant appears HEALTHY! No treatment required.")

    st.markdown("---")

    # Treatment tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Chemical Treatment",
        "Organic / Natural",
        "Cultural Practices",
        "Prevention"
    ])

    with tab1:
        st.markdown(f"**Chemical Treatment:**")
        st.write(info["chemical"])

    with tab2:
        st.markdown(f"**Organic / Natural Treatment:**")
        st.write(info["organic"])

    with tab3:
        st.markdown(f"**Cultural Practices:**")
        st.write(info["cultural"])

    with tab4:
        st.markdown(f"**Prevention Measures:**")
        st.write(info["prevention"])

def pest_detection_page():
    st.title("Pest & Disease Detection")
    st.markdown("Upload a clear, well-lit image of a plant leaf to detect diseases.")

    farmers = get_all_farmers()
    farmer_map = {f"[{row[0]}] {row[1]}": row[0] for row in farmers} if farmers else {}

    selected_fid = None
    if farmer_map:
        sel = st.selectbox("Link to Farmer (optional)", ["-- Skip --"] + list(farmer_map.keys()))
        if sel != "-- Skip --":
            selected_fid = farmer_map[sel]

    uploaded = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", width=350)

        if st.button("Detect Disease", type="primary"):
            with st.spinner("Analyzing image with CNN model..."):
                try:
                    disease, confidence, all_preds = predict_disease(image)
                    clean_name = disease.replace("___", " → ").replace("_", " ")

                    st.subheader(f"Result: {clean_name}")
                    show_solution_card(disease, confidence)

                    # Top 3 predictions
                    with st.expander("See Top 3 Predictions"):
                        top3_idx = np.argsort(all_preds)[-3:][::-1]
                        for idx in top3_idx:
                            name = CLASS_INDICES.get(str(idx), "Unknown").replace("___", " → ").replace("_", " ")
                            prob = all_preds[idx] * 100
                            st.write(f"- **{name}**: {prob:.1f}%")

                    # Save to DB
                    if selected_fid:
                        info = DISEASE_INFO.get(disease, {})
                        solution_summary = info.get("chemical", "See treatment tabs above.")
                        add_pest_log(selected_fid, uploaded.name, disease, confidence, solution_summary)
                        st.caption("Detection logged to farmer profile.")

                except Exception as e:
                    st.error(f"Detection error: {str(e)}")