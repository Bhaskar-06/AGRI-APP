import streamlit as st
import joblib
import numpy as np
from database.db import add_soil_record, get_soil_by_farmer, get_all_farmers
import pandas as pd

FERTILIZER_GUIDE = {
    "rice":     "Use Urea (N), SSP (P), MOP (K). Apply in split doses.",
    "wheat":    "Apply DAP at sowing. Top-dress with Urea at tillering.",
    "maize":    "High N requirement. Use NPK 20:20:0 + Urea.",
    "chickpea": "Low N needed (fixes own). Apply Rhizobium + SSP.",
    "cotton":   "Apply NPK 24:12:12. Foliar spray of boron recommended.",
    "sugarcane":"Heavy feeder — apply FYM + NPK 250:85:100 kg/ha.",
    "tomato":   "Use NPK 19:19:19 at transplant. K-rich at fruiting stage.",
    "potato":   "Apply NPK 120:60:100. Potassium critical for tuber quality.",
    "mango":    "Apply NPK 1:0.5:1 kg/tree. Micronutrients Zn, B important.",
    "banana":   "High K crop. Use NPK 200:60:220 g/plant.",
}

@st.cache_resource
def load_crop_model():
    return joblib.load("models/crop_recommender.pkl")

def soil_health_page():
    st.title("🧪 Soil Health & Crop Recommendation")
    farmers = get_all_farmers()
    farmer_map = {f"[{row[0]}] {row[1]}": row[0] for row in farmers} if farmers else {}

    tab1, tab2 = st.tabs(["Soil Input & Recommend", "Soil History"])

    with tab1:
        with st.form("soil_form"):
            if farmer_map:
                selected = st.selectbox("Select Farmer", list(farmer_map.keys()))
            field = st.text_input("Field Name")
            
            col1, col2 = st.columns(2)
            with col1:
                N    = st.number_input("Nitrogen (N) kg/ha",   0.0, 200.0, 50.0)
                P    = st.number_input("Phosphorus (P) kg/ha", 0.0, 200.0, 30.0)
                K    = st.number_input("Potassium (K) kg/ha",  0.0, 200.0, 40.0)
                ph   = st.slider("Soil pH", 3.0, 10.0, 6.5)
            with col2:
                temp     = st.number_input("Temperature (°C)",  10.0, 50.0, 25.0)
                humidity = st.number_input("Humidity (%)",       10.0, 100.0, 60.0)
                moisture = st.number_input("Moisture (%)",       0.0,  100.0, 40.0)
                rainfall = st.number_input("Rainfall (mm/yr)",  0.0, 3000.0, 500.0)
            
            submit = st.form_submit_button("🔍 Get Recommendation")

        if submit:
            try:
                model = load_crop_model()
                features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
                crop = model.predict(features)[0]
                
                st.success(f"✅ **Recommended Crop: {crop.upper()}**")
                
                # pH advice
                if ph < 5.5:
                    st.warning("⚠️ Soil is too acidic. Apply agricultural lime (CaCO₃) to raise pH.")
                elif ph > 8.0:
                    st.warning("⚠️ Soil is alkaline. Apply gypsum or sulfur to lower pH.")
                else:
                    st.info("✅ Soil pH is in optimal range.")

                # Fertilizer advice
                fertilizer = FERTILIZER_GUIDE.get(crop.lower(), 
                    f"Apply balanced NPK fertilizer. Consult local agronomist for {crop}.")
                st.info(f"**🌿 Fertilizer Recommendation:**\n\n{fertilizer}")
                
                # Save to DB
                if farmer_map:
                    fid = farmer_map[selected]
                    add_soil_record(fid, field, N, P, K, ph, moisture, temp, humidity, rainfall)
                    st.caption("✅ Soil record saved.")
                    
            except Exception as e:
                st.error(f"Model error: {e}. Run train_soil_model.py first.")

    with tab2:
        if farmer_map:
            sel2 = st.selectbox("Select Farmer", list(farmer_map.keys()), key="soil_view")
            fid2 = farmer_map[sel2]
            records = get_soil_by_farmer(fid2)
            if records:
                df = pd.DataFrame(records, columns=[
                    "ID","FarmerID","Field","N","P","K","pH","Moisture","Temp","Humidity","Rainfall","Date"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No soil records yet.")