import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from database.db import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")

@st.cache_resource
def load_recommender():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def soil_health_page():
    st.title("🧪 Soil Health & Crop Recommendation")

    conn = get_connection()
    farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
    conn.close()

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    tab1, tab2 = st.tabs(["Soil Input & Recommend", "Soil History"])

    with tab1:
        st.subheader("Enter Soil Parameters")

        farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()))
        field_name = st.text_input("Field Name")

        col1, col2 = st.columns(2)
        with col1:
            N    = st.number_input("Nitrogen (N) kg/ha",    min_value=0.0, value=50.0)
            P    = st.number_input("Phosphorus (P) kg/ha",  min_value=0.0, value=30.0)
            K    = st.number_input("Potassium (K) kg/ha",   min_value=0.0, value=40.0)
            ph   = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
        with col2:
            temp = st.number_input("Temperature (°C)",      min_value=0.0, value=25.0)
            hum  = st.number_input("Humidity (%)",          min_value=0.0, max_value=100.0, value=60.0)
            moi  = st.number_input("Moisture (%)",          min_value=0.0, max_value=100.0, value=40.0)
            rain = st.number_input("Rainfall (mm/yr)",      min_value=0.0, value=500.0)

        if st.button("🔍 Get Recommendation", type="primary"):
            try:
                model = load_recommender()
                features = np.array([[N, P, K, temp, hum, ph, rain]])
                recommended_crop = model.predict(features)[0]

                st.success(f"### 🌾 Recommended Crop: **{recommended_crop.upper()}**")

                # Soil health score
                score = 0
                tips = []
                if 6.0 <= ph <= 7.5:   score += 30
                else: tips.append("⚠️ Adjust soil pH to 6.0–7.5 range")
                if N >= 40:             score += 20
                else: tips.append("⚠️ Nitrogen is low — apply urea or compost")
                if P >= 20:             score += 20
                else: tips.append("⚠️ Phosphorus is low — apply DAP fertilizer")
                if K >= 20:             score += 20
                else: tips.append("⚠️ Potassium is low — apply MOP fertilizer")
                if hum >= 40:           score += 10
                else: tips.append("⚠️ Humidity is low — consider irrigation")

                st.metric("Soil Health Score", f"{score}/100")
                if score >= 80: st.success("✅ Excellent soil condition!")
                elif score >= 50: st.warning("🟡 Moderate — some improvements needed")
                else: st.error("🔴 Poor soil condition — needs treatment")

                for tip in tips:
                    st.markdown(f"- {tip}")

                # Save to DB
                fid = farmer_options[farmer_key]
                conn = get_connection()
                conn.execute("""
                    INSERT INTO soil_records
                    (farmer_id, field_name, nitrogen, phosphorus, potassium, ph,
                     temperature, humidity, moisture, rainfall, recommended_crop)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (fid, field_name, N, P, K, ph, temp, hum, moi, rain, recommended_crop))
                conn.commit()
                conn.close()
                st.info("📝 Record saved to database.")

            except Exception as e:
                st.error(f"Model error: {e}. Ensure crop_recommender.pkl is in the models/ folder.")

    with tab2:
        st.subheader("Soil Health History")
        conn = get_connection()
        rows = conn.execute("""
            SELECT s.id, COALESCE(f.name,'Guest') as farmer, s.field_name,
                   s.nitrogen, s.phosphorus, s.potassium, s.ph,
                   s.temperature, s.humidity, s.recommended_crop, s.recorded_at
            FROM soil_records s
            LEFT JOIN farmers f ON s.farmer_id = f.id
            ORDER BY s.recorded_at DESC
        """).fetchall()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=["ID","Farmer","Field","N","P","K","pH","Temp","Humidity","Recommended Crop","Date"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No soil records yet.")