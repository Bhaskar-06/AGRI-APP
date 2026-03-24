import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from database.db import add_pest_log, get_all_farmers

# 38 classes from PlantVillage dataset
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

DISEASE_SOLUTIONS = {
    "Apple___Apple_scab": "Apply fungicides like Captan or Mancozeb. Remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicide.",
    "Corn_(maize)___Common_rust_": "Apply propiconazole-based fungicide. Plant resistant varieties.",
    "Potato___Early_blight": "Use chlorothalonil or mancozeb. Ensure proper crop rotation.",
    "Potato___Late_blight": "Apply metalaxyl fungicide immediately. Remove infected plants.",
    "Tomato___Early_blight": "Apply copper fungicide. Maintain proper plant spacing.",
    "Tomato___Late_blight": "Use metalaxyl + mancozeb. Destroy infected plant material.",
    "Tomato___Bacterial_spot": "Apply copper bactericide. Avoid overhead irrigation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Remove infected plants.",
    "Tomato___Leaf_Mold": "Improve ventilation, apply fungicide like chlorothalonil.",
    "Grape___Black_rot": "Remove mummified fruits. Apply myclobutanil fungicide.",
    "Squash___Powdery_mildew": "Apply sulfur-based fungicide or neem oil spray.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply potassium bicarbonate or sulfur sprays.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply captan fungicide.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure — remove infected trees. Control psyllid insects.",
}

@st.cache_resource
def load_model():
    # Load your saved model from models/ folder
    model = tf.keras.models.load_model("models/plant_disease_model.h5")
    return model

def predict_disease(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_idx = np.argmax(predictions)
    return CLASS_NAMES[class_idx], confidence

def pest_detection_page():
    st.title("🔍 Pest & Disease Detection")
    farmers = get_all_farmers()
    farmer_map = {f"[{row[0]}] {row[1]}": row[0] for row in farmers} if farmers else {}

    if farmer_map:
        selected = st.selectbox("Link Detection to Farmer (optional)", ["-- Skip --"] + list(farmer_map.keys()))
    
    uploaded = st.file_uploader("📷 Upload Plant Leaf Image", type=["jpg","jpeg","png"])
    
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("🔬 Detect Disease"):
            with st.spinner("Analyzing image with CNN model..."):
                try:
                    model = load_model()
                    disease, confidence = predict_disease(image, model)
                    
                    st.success(f"**Detected:** `{disease.replace('_', ' ')}`")
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    solution = DISEASE_SOLUTIONS.get(disease, 
                        "Consult local agricultural extension officer. Monitor crop closely.")
                    
                    st.info(f"**💊 Recommended Treatment:**\n\n{solution}")
                    
                    # Save to DB
                    if farmer_map and selected != "-- Skip --":
                        fid = farmer_map[selected]
                        add_pest_log(fid, uploaded.name, disease, confidence, solution)
                        st.caption("✅ Detection logged to farmer profile.")
                        
                except Exception as e:
                    st.error(f"Model error: {e}. Ensure plant_disease_model.h5 is in models/ folder.")