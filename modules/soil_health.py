import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from database.db import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

@st.cache_resource
def load_recommender():
    # Try loading pkl first; if it fails, retrain from CSV
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        # Quick test to verify it works
        model.predict([[50, 30, 40, 25, 60, 6.5, 500]])
        return model
    except Exception:
        pass  # Fall through to retraining

    # Retrain from CSV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(CSV_PATH)
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    feature_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols].values
    y = df['label'].values

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save retrained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    return clf

def soil_health_page():
    st.title("🧪 Soil Health & Crop Recommendation")

    try:
        conn = get_connection()
        farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    tab1, tab2 = st.tabs(["Soil Input & Recommend", "Soil History"])

    with tab1:
        st.subheader("Enter Soil Parameters")
        farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="sh_farmer")
        field_name = st.text_input("Field Name", key="sh_field")

        col1, col2 = st.columns(2)
        with col1:
            N    = st.number_input("Nitrogen (N) kg/ha",   min_value=0.0, max_value=200.0, value=50.0, key="sh_n")
            P    = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=200.0, value=30.0, key="sh_p")
            K    = st.number_input("Potassium (K) kg/ha",  min_value=0.0, max_value=200.0, value=40.0, key="sh_k")
            ph   = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, key="sh_ph")
        with col2:
            temp = st.number_input("Temperature (°C)",     min_value=0.0, max_value=60.0,  value=25.0, key="sh_temp")
            hum  = st.number_input("Humidity (%)",         min_value=0.0, max_value=100.0, value=60.0, key="sh_hum")
            moi  = st.number_input("Moisture (%)",         min_value=0.0, max_value=100.0, value=40.0, key="sh_moi")
            rain = st.number_input("Rainfall (mm/yr)",     min_value=0.0, max_value=3000.0,value=500.0,key="sh_rain")

        if st.button("🔍 Get Recommendation", type="primary", key="sh_submit"):
            with st.spinner("Loading AI model and analyzing soil..."):
                try:
                    model = load_recommender()
                    features = np.array([[N, P, K, temp, hum, ph, rain]])
                    recommended_crop = model.predict(features)[0]

                    st.success(f"## 🌾 Recommended Crop: **{str(recommended_crop).upper()}**")

                    # Soil Health Score
                    score = 0
                    tips  = []
                    if 6.0 <= ph <= 7.5:  score += 30
                    else: tips.append("⚠️ Adjust soil pH to 6.0–7.5")
                    if N >= 40:            score += 20
                    else: tips.append("⚠️ Low Nitrogen — apply urea or compost")
                    if P >= 20:            score += 20
                    else: tips.append("⚠️ Low Phosphorus — apply DAP fertilizer")
                    if K >= 20:            score += 20
                    else: tips.append("⚠️ Low Potassium — apply MOP fertilizer")
                    if hum >= 40:          score += 10
                    else: tips.append("⚠️ Low Humidity — consider irrigation")

                    st.metric("Soil Health Score", f"{score}/100")
                    if score >= 80:   st.success("✅ Excellent soil condition!")
                    elif score >= 50: st.warning("🟡 Moderate — improvements needed")
                    else:             st.error("🔴 Poor soil — needs treatment")

                    for tip in tips:
                        st.markdown(f"- {tip}")

                    # Save to DB
                    fid = farmer_options[farmer_key]
                    conn = get_connection()
                    conn.execute("""
                        INSERT INTO soil_records
                        (farmer_id, field_name, nitrogen, phosphorus, potassium,
                         ph, temperature, humidity, moisture, rainfall, recommended_crop)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (fid, field_name, N, P, K, ph, temp, hum, moi, rain, str(recommended_crop)))
                    conn.commit()
                    conn.close()
                    st.info("📝 Record saved to database.")

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Soil Health History")
        try:
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
        except Exception as e:
            st.error(f"Database error: {e}")