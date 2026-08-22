import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from database.db import db_add_soil, db_get_soil_records, db_get_farmers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")  # matches actual filename (capital C)


@st.cache_resource(show_spinner=False)
def load_recommender():
    """Load the trained model, or train a fresh one from the CSV if the
    pickle is missing/incompatible."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model.predict([[50, 30, 40, 25, 60, 6.5, 500]])  # sanity check
        return model
    except Exception:
        pass

    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    X = df[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']].values
    y = df['label'].values
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    return clf


def soil_health_page():
    st.title("🧪 Soil Health & Crop Recommendation")

    if "soil_result" not in st.session_state:
        st.session_state["soil_result"] = None

    farmers = db_get_farmers()
    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{f['id']} - {f['name']}": f['id'] for f in farmers})

    tab1, tab2 = st.tabs(["Soil Input & Recommend", "Soil History"])

    with tab1:
        st.subheader("Enter Soil Parameters")
        farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="sh_farmer")
        field_name = st.text_input("Field Name", key="sh_field")

        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N) kg/ha", 0.0, 200.0, 50.0, key="sh_n")
            P = st.number_input("Phosphorus (P) kg/ha", 0.0, 200.0, 30.0, key="sh_p")
            K = st.number_input("Potassium (K) kg/ha", 0.0, 200.0, 40.0, key="sh_k")
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, key="sh_ph")
        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="sh_temp")
            hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="sh_hum")
            moi = st.number_input("Moisture (%)", 0.0, 100.0, 40.0, key="sh_moi")
            rain = st.number_input("Rainfall (mm/yr)", 0.0, 3000.0, 500.0, key="sh_rain")

        if st.button("🔍 Get Recommendation", type="primary", key="sh_submit"):
            with st.spinner("Training/loading AI model and analyzing soil data..."):
                try:
                    model = load_recommender()
                    features = np.array([[N, P, K, temp, hum, ph, rain]])
                    recommended_crop = str(model.predict(features)[0])

                    score, tips = 0, []
                    if 6.0 <= ph <= 7.5:
                        score += 30
                    else:
                        tips.append("⚠️ Adjust soil pH to 6.0–7.5 (lime for acidic, sulfur for alkaline)")
                    if N >= 40:
                        score += 20
                    else:
                        tips.append("⚠️ Low Nitrogen — apply Urea 46% @ 100kg/acre or compost")
                    if P >= 20:
                        score += 20
                    else:
                        tips.append("⚠️ Low Phosphorus — apply DAP fertilizer @ 50kg/acre")
                    if K >= 20:
                        score += 20
                    else:
                        tips.append("⚠️ Low Potassium — apply MOP (Muriate of Potash) @ 50kg/acre")
                    if hum >= 40:
                        score += 10
                    else:
                        tips.append("⚠️ Low Humidity — increase irrigation frequency")

                    st.session_state["soil_result"] = {
                        "recommended_crop": recommended_crop,
                        "score": score,
                        "tips": tips,
                    }

                    db_add_soil(
                        farmer_id=farmer_options[farmer_key],
                        field_name=field_name,
                        nitrogen=N,
                        phosphorus=P,
                        potassium=K,
                        ph=ph,
                        temperature=temp,
                        humidity=hum,
                        moisture=moi,
                        rainfall=rain,
                        recommended_crop=recommended_crop,
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state["soil_result"]:
            r = st.session_state["soil_result"]
            st.markdown("---")
            st.success(f"## 🌾 Recommended Crop: **{r['recommended_crop'].upper()}**")
            st.metric("Soil Health Score", f"{r['score']}/100")
            if r['score'] >= 80:
                st.success("✅ Excellent soil condition!")
            elif r['score'] >= 50:
                st.warning("🟡 Moderate — some improvements needed")
            else:
                st.error("🔴 Poor soil condition — urgent treatment needed")
            for tip in r["tips"]:
                st.markdown(f"- {tip}")

            if st.button("🔄 Clear & Analyze New Soil", key="sh_clear"):
                st.session_state["soil_result"] = None
                st.rerun()

    with tab2:
        st.subheader("Soil Health History")
        rows = db_get_soil_records()
        if rows:
            df = pd.DataFrame([{
                "ID": r.get("id", ""),
                "Farmer": (r.get("farmers") or {}).get("name", "Guest"),
                "Field": r.get("field_name", ""),
                "N": r.get("nitrogen", ""),
                "P": r.get("phosphorus", ""),
                "K": r.get("potassium", ""),
                "pH": r.get("ph_level", ""),
                "Temp": r.get("temperature", ""),
                "Humidity": r.get("humidity", ""),
                "Rainfall": r.get("rainfall", ""),
                "Recommended Crop": r.get("recommended_crop", ""),
                "Date": str(r.get("created_at", ""))[:10],
            } for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "soil_history.csv", "text/csv", use_container_width=True)
        else:
            st.info("No soil records yet.")