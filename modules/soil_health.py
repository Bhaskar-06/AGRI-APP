import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
from database.db import get_connection

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

@st.cache_resource
def load_recommender():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model.predict([[50,30,40,25,60,6.5,500]])
        return model
    except Exception:
        pass
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    X = df[['n','p','k','temperature','humidity','ph','rainfall']].values
    y = df['label'].values
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    return clf

def soil_health_page():
    st.title("🧪 Soil Health & Crop Recommendation")

    # Initialize session state so results don't disappear
    if "soil_result" not in st.session_state:
        st.session_state["soil_result"] = None

    conn    = get_connection()
    farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
    conn.close()

    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{r['id']} - {r['name']}": r['id'] for r in farmers})

    tab1, tab2 = st.tabs(["Soil Input & Recommend", "Soil History"])

    with tab1:
        st.subheader("Enter Soil Parameters")
        farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="sh_farmer")
        field_name = st.text_input("Field Name", key="sh_field")

        col1, col2 = st.columns(2)
        with col1:
            N    = st.number_input("Nitrogen (N) kg/ha",   0.0, 200.0, 50.0,  key="sh_n")
            P    = st.number_input("Phosphorus (P) kg/ha", 0.0, 200.0, 30.0,  key="sh_p")
            K    = st.number_input("Potassium (K) kg/ha",  0.0, 200.0, 40.0,  key="sh_k")
            ph   = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1,                  key="sh_ph")
        with col2:
            temp = st.number_input("Temperature (°C)",     0.0, 60.0,  25.0,  key="sh_temp")
            hum  = st.number_input("Humidity (%)",         0.0, 100.0, 60.0,  key="sh_hum")
            moi  = st.number_input("Moisture (%)",         0.0, 100.0, 40.0,  key="sh_moi")
            rain = st.number_input("Rainfall (mm/yr)",     0.0, 3000.0,500.0, key="sh_rain")

        if st.button("🔍 Get Recommendation", type="primary", key="sh_submit"):
            with st.spinner("Training/loading AI model and analyzing soil data..."):
                try:
                    model            = load_recommender()
                    features         = np.array([[N, P, K, temp, hum, ph, rain]])
                    recommended_crop = str(model.predict(features)[0])
                    score, tips      = 0, []
                    if 6.0 <= ph <= 7.5: score += 30
                    else: tips.append("⚠️ Adjust soil pH to 6.0–7.5 (lime for acidic, sulfur for alkaline)")
                    if N >= 40:  score += 20
                    else: tips.append("⚠️ Low Nitrogen — apply Urea 46% @ 100kg/acre or compost")
                    if P >= 20:  score += 20
                    else: tips.append("⚠️ Low Phosphorus — apply DAP fertilizer @ 50kg/acre")
                    if K >= 20:  score += 20
                    else: tips.append("⚠️ Low Potassium — apply MOP (Muriate of Potash) @ 50kg/acre")
                    if hum >= 40: score += 10
                    else: tips.append("⚠️ Low Humidity — increase irrigation frequency")

                    # Store in session_state
                    st.session_state["soil_result"] = {
                        "recommended_crop": recommended_crop,
                        "score": score, "tips": tips,
                        "fid": farmer_options[farmer_key],
                        "field_name": field_name,
                        "N":N,"P":P,"K":K,"ph":ph,"temp":temp,"hum":hum,"moi":moi,"rain":rain
                    }
                    # Save to DB
                    conn = get_connection()
                    conn.execute("""INSERT INTO soil_records
                        (farmer_id,field_name,nitrogen,phosphorus,potassium,ph,temperature,humidity,moisture,rainfall,recommended_crop)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (farmer_options[farmer_key], field_name, N, P, K, ph, temp, hum, moi, rain, recommended_crop))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Always render result from session_state (won't disappear)
        if st.session_state["soil_result"]:
            r = st.session_state["soil_result"]
            st.markdown("---")
            st.success(f"## 🌾 Recommended Crop: **{r['recommended_crop'].upper()}**")
            st.metric("Soil Health Score", f"{r['score']}/100")
            if r['score'] >= 80:   st.success("✅ Excellent soil condition!")
            elif r['score'] >= 50: st.warning("🟡 Moderate — some improvements needed")
            else:                  st.error("🔴 Poor soil condition — urgent treatment needed")
            for tip in r["tips"]:
                st.markdown(f"- {tip}")

            if st.button("🔄 Clear & Analyze New Soil", key="sh_clear"):
                st.session_state["soil_result"] = None
                st.rerun()

    with tab2:
        st.subheader("Soil Health History")
        try:
            conn  = get_connection()
            rows  = conn.execute("""
                SELECT s.id, COALESCE(f.name,'Guest'), s.field_name, s.nitrogen,
                       s.phosphorus, s.potassium, s.ph, s.temperature,
                       s.humidity, s.recommended_crop, s.recorded_at
                FROM soil_records s LEFT JOIN farmers f ON s.farmer_id=f.id
                ORDER BY s.recorded_at DESC""").fetchall()
            conn.close()
            if rows:
                df = pd.DataFrame(rows, columns=["ID","Farmer","Field","N","P","K","pH","Temp","Humidity","Recommended Crop","Date"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No soil records yet.")
        except Exception as e:
            st.error(f"Database error: {e}")