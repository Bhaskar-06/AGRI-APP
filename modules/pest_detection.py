import os, json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from database.db import get_connection

BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH         = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

@st.cache_resource
def load_disease_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        indices = json.load(f)
    return {v: k for k, v in indices.items()}

# ── Master Disease Solutions (covers ALL PlantVillage classes + extended crops)
DISEASE_SOLUTIONS = {
    # APPLE
    "Apple___Apple_scab":           {"type":"Fungal","severity":"Medium","chemical":"Apply Captan 50WP @ 2g/L or Mancozeb 75WP every 7-10 days.","organic":"Neem oil 5ml/L spray weekly. Remove and destroy fallen leaves.","cultural":"Prune for airflow. Avoid overhead irrigation.","prevention":"Rake fallen leaves in autumn. Plant scab-resistant varieties."},
    "Apple___Black_rot":            {"type":"Fungal","severity":"High","chemical":"Apply Thiophanate-methyl 70WP @ 1g/L or Captan fungicide.","organic":"Copper sulfate spray (Bordeaux mixture 1%).","cultural":"Remove mummified fruits and all dead wood immediately.","prevention":"Maintain tree vigor. Sanitize pruning tools."},
    "Apple___Cedar_apple_rust":     {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil or Propiconazole at bud break (pink stage).","organic":"Sulfur-based fungicide spray every 7 days.","cultural":"Remove nearby juniper/cedar trees within 300m if possible.","prevention":"Plant rust-resistant apple varieties."},
    "Apple___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive neem oil spray once a month.","cultural":"Maintain good pruning. Remove water sprouts.","prevention":"Scout every 2 weeks. Keep records of any abnormalities."},
    # BLUEBERRY
    "Blueberry___healthy":          {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Mulch with pine bark to maintain acidic pH.","cultural":"Maintain soil pH 4.5–5.5. Prune old canes.","prevention":"Scout for mummy berry disease regularly."},
    # CHERRY
    "Cherry_(including_sour)___Powdery_mildew": {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil 10WP @ 1g/L or Trifloxystrobin.","organic":"Baking soda 5g/L + neem oil 5ml/L weekly spray.","cultural":"Prune infected shoot tips immediately. Improve air circulation.","prevention":"Avoid excessive nitrogen fertilizer. Choose resistant varieties."},
    "Cherry_(including_sour)___healthy": {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive copper spray in early spring.","cultural":"Thin fruits for better airflow. Remove dead wood.","prevention":"Monitor for cherry leaf spot and brown rot."},
    # CORN
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin 23SC @ 1ml/L or Propiconazole 25EC.","organic":"Copper hydroxide spray @ 3g/L.","cultural":"Crop rotation every 2 years. Reduce plant density.","prevention":"Use resistant hybrids. Avoid overhead irrigation."},
    "Corn_(maize)___Common_rust_":  {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin + Propiconazole (Headline AMP) at first sign.","organic":"Sulfur dust 80WP @ 3kg/ha.","cultural":"Plant early-maturing varieties. Avoid late planting.","prevention":"Monitor during warm humid weather (20-25°C)."},
    "Corn_(maize)___Northern_Leaf_Blight": {"type":"Fungal","severity":"High","chemical":"Apply Azoxystrobin + Propiconazole at tasseling stage.","organic":"Bacillus subtilis biofungicide spray.","cultural":"Crop rotation with soybean or wheat. Remove crop debris.","prevention":"Plant resistant hybrids. Ensure proper plant spacing."},
    "Corn_(maize)___healthy":       {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Maintain 60cm row spacing. Side-dress with nitrogen at V6.","prevention":"Scout fields weekly during critical growth stages."},
    # GRAPE
    "Grape___Black_rot":            {"type":"Fungal","severity":"High","chemical":"Apply Mancozeb 75WP @ 2.5g/L before and after bloom.","organic":"Copper sulfate spray (Bordeaux mixture).","cultural":"Remove all mummified berries and canes. Improve canopy airflow.","prevention":"Prune properly in winter. Avoid wetting foliage."},
    "Grape___Esca_(Black_Measles)": {"type":"Fungal","severity":"High","chemical":"Protect pruning wounds immediately with Thiophanate-methyl paste.","organic":"Trichoderma harzianum biological paste on wounds.","cultural":"Remove and burn infected wood. Sterilize all pruning tools with 10% bleach.","prevention":"Make smaller pruning cuts. Always protect wounds in wet weather."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"type":"Fungal","severity":"Medium","chemical":"Apply Mancozeb 75WP or Copper oxychloride 50WP @ 3g/L.","organic":"Neem oil 5ml/L + copper soap spray weekly.","cultural":"Improve trellis ventilation. Remove and burn infected leaves.","prevention":"Avoid overhead irrigation. Mulch to prevent soil splash."},
    "Grape___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost mulch. Monthly neem oil preventive spray.","cultural":"Remove old fruiting canes after harvest. Tie new canes properly.","prevention":"Monitor for downy mildew and botrytis bunch rot."},
    # ORANGE
    "Orange___Haunglongbing_(Citrus_greening)": {"type":"Bacterial","severity":"Very High","chemical":"Control Asian citrus psyllid with Imidacloprid 70WG @ 0.5g/L systemically.","organic":"Neem oil + pyrethrin spray for psyllid control. Remove infected trees.","cultural":"Immediately remove and destroy all infected trees. Do not move plant material.","prevention":"Use certified disease-free nursery budwood. Install psyllid monitoring yellow sticky traps."},
    # PEACH
    "Peach___Bacterial_spot":       {"type":"Bacterial","severity":"High","chemical":"Apply Oxytetracycline 3.4% spray or Copper hydroxide 53.8DF @ 3g/L.","organic":"Copper hydroxide spray in early spring.","cultural":"Prune to improve airflow. Avoid overhead irrigation. Remove infected leaves.","prevention":"Plant resistant peach/nectarine varieties. Avoid wounding bark."},
    "Peach___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Preventive copper spray at bud swell in spring.","cultural":"Thin fruits to 15-20cm apart. Remove mummified fruits.","prevention":"Monitor for brown rot and peach leaf curl."},
    # PEPPER
    "Pepper,_bell___Bacterial_spot":{"type":"Bacterial","severity":"High","chemical":"Apply Copper bactericide + Mancozeb 75WP @ 2g/L. Avoid application in rain.","organic":"Copper soap spray 10ml/L. Remove and destroy infected leaves immediately.","cultural":"Use drip irrigation only. Stake plants to improve airflow.","prevention":"Use certified disease-free transplants. Rotate pepper fields every 3 years."},
    "Pepper,_bell___healthy":       {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Monthly neem oil preventive spray.","cultural":"Proper staking. Side-dress with calcium nitrate at flowering.","prevention":"Scout weekly for aphids and whiteflies as virus vectors."},
    # POTATO
    "Potato___Early_blight":        {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil 75WP @ 2g/L or Mancozeb 75WP every 7 days.","organic":"Copper-based fungicide or baking soda 5g/L spray.","cultural":"Remove lower infected leaves. Avoid overhead irrigation after noon.","prevention":"Use certified disease-free seed potatoes. Rotate crops every 3 years."},
    "Potato___Late_blight":         {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl + Mancozeb (Ridomil Gold MZ) @ 2.5g/L immediately. Repeat every 5 days.","organic":"Copper hydroxide 53.8DF @ 3g/L spray. Remove and destroy all infected plants.","cultural":"Destroy volunteer plants. Avoid all leaf wetness. Hill soil high around stems.","prevention":"Plant resistant varieties (Sarpo Mira, Defender). Monitor daily during cool wet weather."},
    "Potato___healthy":             {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost application to improve soil health.","cultural":"Hill soil around plants at 20cm height. Ensure good drainage.","prevention":"Rotate crops every 3 years. Use certified seed potatoes only."},
    # RASPBERRY
    "Raspberry___healthy":          {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Neem oil spray monthly.","cultural":"Remove all old floricanes after harvest. Thin primocanes to 15cm spacing.","prevention":"Scout for cane diseases and botrytis fruit rot."},
    # RICE
    "Rice___Bacterial_leaf_blight": {"type":"Bacterial","severity":"High","chemical":"Apply Copper oxychloride 50WP @ 3g/L or Bismerthiazol 25WP.","organic":"Pseudomonas fluorescens bioagent @ 5g/L spray.","cultural":"Drain flooded fields. Reduce nitrogen. Avoid mechanical injury to plants.","prevention":"Use resistant varieties (IR64, BPT-5204). Treat seeds with hot water at 52°C for 30 min."},
    "Rice___Brown_spot":            {"type":"Fungal","severity":"Medium","chemical":"Apply Edifenphos 50EC @ 1ml/L or Tricyclazole 75WP @ 0.6g/L.","organic":"Neem seed kernel extract 5% spray.","cultural":"Balanced NPK fertilization. Avoid water stress or waterlogging.","prevention":"Use resistant varieties. Treat seeds with Carbendazim 50WP before sowing."},
    "Rice___Leaf_smut":             {"type":"Fungal","severity":"Low","chemical":"Seed treatment with Carbendazim 50WP @ 2g/kg seed.","organic":"Trichoderma viride seed treatment @ 4g/kg seed.","cultural":"Deep summer plowing. Remove and burn infected stubble after harvest.","prevention":"Use certified clean seeds. Practice crop rotation with legumes."},
    "Rice___healthy":               {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Maintain 5cm water level. Apply nitrogen in split doses.","prevention":"Regular field scouting every 7 days."},
    # SOYBEAN
    "Soybean___healthy":            {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Rhizobium japonicum inoculant at planting for nitrogen fixation.","cultural":"Maintain 45cm row spacing. Ensure good soil drainage.","prevention":"Monitor for sudden death syndrome and white mold."},
    # SQUASH
    "Squash___Powdery_mildew":      {"type":"Fungal","severity":"Medium","chemical":"Apply Trifloxystrobin 50WG @ 0.5g/L or Tebuconazole 25EC @ 1ml/L at first sign.","organic":"Baking soda 5g/L + neem oil 5ml/L + dish soap 1ml/L weekly spray.","cultural":"Improve air circulation between plants. Remove infected leaves immediately.","prevention":"Plant resistant varieties. Avoid excessive nitrogen fertilizer."},
    # STRAWBERRY
    "Strawberry___Leaf_scorch":     {"type":"Fungal","severity":"Medium","chemical":"Apply Myclobutanil 10WP @ 1g/L or Captan 50WP @ 2g/L.","organic":"Copper-based spray weekly during humid periods.","cultural":"Remove infected leaves. Avoid overhead irrigation.","prevention":"Plant certified virus-free transplants. Maintain proper plant spacing."},
    "Strawberry___healthy":         {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost mulch with straw. Monthly neem oil spray.","cultural":"Remove old leaves and runners after harvest. Renovate bed.","prevention":"Monitor for grey mold (Botrytis) and spider mites."},
    # TOMATO
    "Tomato___Bacterial_spot":      {"type":"Bacterial","severity":"High","chemical":"Apply Copper hydroxide 53.8DF + Mancozeb 75WP @ 2g/L each. Apply before rain.","organic":"Copper soap spray 10ml/L. Remove infected leaves immediately.","cultural":"Use drip irrigation only. Stake plants to keep foliage dry.","prevention":"Use certified disease-free transplants. Rotate fields every 3 years."},
    "Tomato___Early_blight":        {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil 75WP @ 2g/L or Mancozeb 75WP every 7-10 days from first symptom.","organic":"Copper fungicide spray. Mulch heavily around plants.","cultural":"Remove lower leaves touching soil. Avoid wetting foliage.","prevention":"Rotate tomatoes every 2-3 years. Use disease-resistant varieties."},
    "Tomato___Late_blight":         {"type":"Fungal/Oomycete","severity":"Very High","chemical":"Apply Metalaxyl 8% + Mancozeb 64% (Ridomil) @ 2.5g/L immediately. Repeat every 5-7 days.","organic":"Copper hydroxide spray @ 3g/L. Remove all infected plant parts immediately.","cultural":"Improve drainage. Avoid all leaf wetness. Remove volunteer plants.","prevention":"Plant Phytophthora-resistant varieties. Never plant near potato fields."},
    "Tomato___Leaf_Mold":           {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil 75WP @ 2g/L or Mancozeb 75WP.","organic":"Baking soda 5g/L + neem oil 5ml/L spray weekly.","cultural":"Reduce greenhouse humidity below 85%. Improve ventilation significantly.","prevention":"Remove infected leaves. Avoid wetting foliage when irrigating."},
    "Tomato___Septoria_leaf_spot":  {"type":"Fungal","severity":"Medium","chemical":"Apply Chlorothalonil 75WP @ 2g/L at first sign of spots. Repeat every 10 days.","organic":"Copper soap spray 10ml/L. Remove infected leaves.","cultural":"Mulch heavily to prevent soil splash. Remove lower leaves proactively.","prevention":"Crop rotation every 3 years. Use disease-free transplants."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"type":"Pest (Mite)","severity":"Medium","chemical":"Apply Abamectin 1.8EC @ 1ml/L or Bifenazate 43SC @ 1ml/L. Rotate miticides.","organic":"Neem oil 5ml/L spray. Introduce Phytoseiulus persimilis predatory mites.","cultural":"Avoid dusty conditions. Overhead water spray knocks off mites.","prevention":"Scout leaf undersides weekly. Keep field borders weed-free."},
    "Tomato___Target_Spot":         {"type":"Fungal","severity":"Medium","chemical":"Apply Azoxystrobin 23SC @ 1ml/L or Boscalid 38.9WG @ 1g/L.","organic":"Copper-based fungicide spray.","cultural":"Remove infected leaves. Improve air circulation between plants.","prevention":"Avoid dense planting. Use mulch to prevent soil splash."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"type":"Viral (Whitefly)","severity":"Very High","chemical":"Control whiteflies with Imidacloprid 70WG @ 0.3g/L soil drench or foliar spray.","organic":"Yellow sticky traps (1/10sqm). Neem oil + pyrethrin spray for whiteflies.","cultural":"Remove and immediately destroy all infected plants. Do not compost them.","prevention":"Use TYLCV-resistant varieties (HM 7883, Shanty F1). Install 40-mesh insect-proof netting."},
    "Tomato___Tomato_mosaic_virus": {"type":"Viral","severity":"High","chemical":"No chemical cure exists. Disinfect all tools with 10% bleach or 70% alcohol.","organic":"Remove and destroy infected plants. Do not compost infected material.","cultural":"Wash hands thoroughly before touching plants. Control aphids as virus vectors.","prevention":"Use ToMV-resistant certified seeds/varieties. Avoid tobacco near plants."},
    "Tomato___healthy":             {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"Compost tea spray for immunity boosting monthly.","cultural":"Maintain proper staking and regular pruning of suckers.","prevention":"Scout weekly for early signs. Keep detailed crop records."},
    # WHEAT
    "Wheat___Brown_rust":           {"type":"Fungal","severity":"High","chemical":"Apply Propiconazole 25EC @ 1ml/L at flag leaf stage. Repeat if needed.","organic":"Sulfur 80WP dust @ 25kg/ha.","cultural":"Avoid late sowing. Apply balanced NPK fertilizer.","prevention":"Use resistant varieties (DBW303, HD3226). Early season monitoring is critical."},
    "Wheat___Yellow_rust":          {"type":"Fungal","severity":"Very High","chemical":"Apply Tebuconazole 25.9EC @ 1ml/L or Propiconazole 25EC immediately at first sign.","organic":"Sulfur-based fungicide spray @ 3kg/ha.","cultural":"Avoid excessive nitrogen application. Plant early to avoid peak infection period.","prevention":"Plant stripe-rust resistant varieties. Monitor fields from February onwards."},
    "Wheat___healthy":              {"type":"None","severity":"None","chemical":"No treatment needed.","organic":"No treatment needed.","cultural":"Balanced NPK + micronutrient fertilization. Proper irrigation at crown root initiation.","prevention":"Regular field monitoring from tillering stage onwards."},
}

# ── Universal fallback solutions by crop type
CROP_GENERIC_SOLUTIONS = {
    "tomato":     {"chemical":"Apply Chlorothalonil 75WP @ 2g/L for fungal issues.","organic":"Neem oil 5ml/L spray weekly.","cultural":"Stake plants. Use drip irrigation.","prevention":"Scout weekly. Rotate crops annually."},
    "potato":     {"chemical":"Apply Mancozeb 75WP @ 2.5g/L preventively.","organic":"Copper-based spray every 10 days.","cultural":"Hill soil. Ensure good drainage.","prevention":"Use certified seed potatoes."},
    "wheat":      {"chemical":"Apply Propiconazole 25EC @ 1ml/L at flag leaf stage.","organic":"Sulfur dust application.","cultural":"Balanced fertilization.","prevention":"Use resistant varieties."},
    "rice":       {"chemical":"Apply Tricyclazole 75WP @ 0.6g/L for blast.","organic":"Pseudomonas fluorescens spray.","cultural":"Balanced nitrogen. Field drainage.","prevention":"Use resistant varieties."},
    "corn":       {"chemical":"Apply Azoxystrobin + Propiconazole at tasseling.","organic":"Bacillus subtilis spray.","cultural":"Crop rotation.","prevention":"Use resistant hybrids."},
    "grape":      {"chemical":"Apply Mancozeb 75WP before and after bloom.","organic":"Bordeaux mixture 1%.","cultural":"Canopy management. Remove infected material.","prevention":"Prune properly. Avoid wetting foliage."},
    "apple":      {"chemical":"Apply Captan 50WP @ 2g/L every 7-10 days.","organic":"Neem oil spray monthly.","cultural":"Prune for airflow.","prevention":"Rake fallen leaves. Plant resistant varieties."},
    "default":    {"chemical":"Consult local agricultural extension officer for specific chemical recommendation.","organic":"Neem oil 5ml/L spray as general preventive measure.","cultural":"Ensure proper plant spacing and airflow. Remove infected plant material.","prevention":"Regular scouting every 7 days. Maintain field hygiene."},
}

def get_solution(raw_name: str) -> dict:
    """Find the best matching solution for a disease class name."""
    # Direct match
    if raw_name in DISEASE_SOLUTIONS:
        return DISEASE_SOLUTIONS[raw_name]
    # Try normalizing separators
    for sep_from, sep_to in [("__", "___"), ("___", "__")]:
        candidate = raw_name.replace(sep_from, sep_to)
        if candidate in DISEASE_SOLUTIONS:
            return DISEASE_SOLUTIONS[candidate]
    # Case-insensitive match
    lower_map = {k.lower(): v for k, v in DISEASE_SOLUTIONS.items()}
    if raw_name.lower() in lower_map:
        return lower_map[raw_name.lower()]
    # Generic fallback by crop type
    crop = raw_name.split("___")[0].split("__")[0].split("_(")[0].lower().strip()
    for known_crop, sol in CROP_GENERIC_SOLUTIONS.items():
        if known_crop in crop:
            return {"type":"Unknown","severity":"Unknown", **sol}
    return {"type":"Unknown","severity":"Unknown", **CROP_GENERIC_SOLUTIONS["default"]}

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict_disease(image: Image.Image):
    model        = load_disease_model()
    idx_to_class = load_class_indices()
    preds        = model.predict(preprocess_image(image))
    idx          = int(np.argmax(preds[0]))
    confidence   = float(np.max(preds[0])) * 100
    raw_name     = idx_to_class.get(idx, "Unknown")
    solution     = get_solution(raw_name)
    return raw_name, confidence, solution

def pest_detection_page():
    st.title("🔍 Pest & Disease Detection")
    st.markdown("Upload a **clear, close-up image** of the affected leaf for best results.")

    # Initialize session state for results
    if "pest_result" not in st.session_state:
        st.session_state["pest_result"] = None

    conn    = get_connection()
    farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
    conn.close()

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    farmer_key    = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="pd_farmer")
    uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg","jpeg","png"], key="pd_upload")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Disease", type="primary", key="pd_detect"):
            with st.spinner("Analyzing with AI model... please wait"):
                raw_name, confidence, solution = predict_disease(image)
                # ── Store in session_state so result NEVER disappears
                st.session_state["pest_result"] = {
                    "raw_name": raw_name,
                    "confidence": confidence,
                    "solution": solution,
                    "farmer_key": farmer_key,
                    "farmer_options": farmer_options
                }
                # Save to DB
                fid = farmer_options[farmer_key]
                crop_part = raw_name.split("___")[0].split("__")[0]
                display = raw_name.replace("___"," → ").replace("__"," → ").replace("_"," ")
                conn = get_connection()
                conn.execute("""INSERT INTO pest_detections (farmer_id,crop_name,disease_detected,confidence,treatment)
                    VALUES (?,?,?,?,?)""", (fid, crop_part, display, confidence, solution["chemical"]))
                conn.commit()
                conn.close()

    # ── Always render result from session_state (won't disappear)
    if st.session_state["pest_result"]:
        r          = st.session_state["pest_result"]
        raw_name   = r["raw_name"]
        confidence = r["confidence"]
        solution   = r["solution"]
        display    = raw_name.replace("___"," → ").replace("__"," → ").replace("_"," ")

        st.markdown("---")
        st.success(f"### ✅ Detected: **{display}**")
        st.metric("Model Confidence", f"{confidence:.1f}%")

        sev_icon = {"None":"🟢","Low":"🟡","Medium":"🟠","High":"🔴","Very High":"🚨"}.get(solution.get("severity",""),"⚪")
        st.markdown(f"### {sev_icon} Severity: **{solution.get('severity','Unknown')}** | Type: **{solution.get('type','Unknown')}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💊 Chemical Treatment")
            st.info(solution.get("chemical","Consult an agricultural expert."))
            st.markdown("#### 🌿 Organic Treatment")
            st.success(solution.get("organic","Use neem oil spray as a general preventive measure."))
        with col2:
            st.markdown("#### 🌾 Cultural Practices")
            st.warning(solution.get("cultural","Ensure proper spacing and remove infected plant material."))
            st.markdown("#### 🛡️ Prevention")
            st.error(solution.get("prevention","Regular scouting every 7 days recommended."))

        if st.button("🔄 Clear Result & Detect New Image", key="pd_clear"):
            st.session_state["pest_result"] = None
            st.rerun()