import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from database.db import db_add_soil, db_get_soil_records, db_get_farmers
from modules.soil_image_detection import soil_image_detection_page

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")


@st.cache_resource(show_spinner=False)
def load_recommender():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        model.predict([[50, 30, 40, 25, 60, 6.5, 500]])
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


def record_soil_tab():
    st.subheader("Enter Soil Parameters")

    farmers = db_get_farmers()
    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{f['id']} - {f['name']}": f['id'] for f in farmers})

    if "soil_result" not in st.session_state:
        st.session_state["soil_result"] = None

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


def view_soil_tab():
    st.subheader("Soil Health History")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="soil_refresh"):
            st.rerun()

    rows = db_get_soil_records()

    if not rows:
        st.info("No soil records yet.")
        return

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


def crop_recommendation_tab():
    st.subheader("🌾 Quick Crop Recommendation")
    st.caption("Based on N, P, K, temperature, humidity, pH and rainfall — no record saved.")

    model = load_recommender()

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0, key="cr_n")
        p = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0, key="cr_p")
        k = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0, key="cr_k")
        temperature = st.number_input("Temperature (°C)", value=25.0, step=0.5, key="cr_temp")

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0, key="cr_hum")
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key="cr_ph")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0, step=1.0, key="cr_rain")

    if st.button("🌾 Recommend Crop", type="primary", use_container_width=True, key="cr_btn"):
        try:
            features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = model.predict(features)[0]
            st.success(f"✅ Recommended Crop: **{str(prediction).title()}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                classes = model.classes_
                top3_idx = proba.argsort()[::-1][:3]
                st.markdown("#### 📊 Top 3 Matches")
                for i in top3_idx:
                    st.write(f"**{str(classes[i]).title()}** — `{proba[i]*100:.1f}%`")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


def soil_health_page():
    st.markdown("# 🧪 Soil Health")
    st.markdown("Log soil test results, scan soil images, and get AI-based crop recommendations.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "➕ Record Soil Data",
        "📋 View Records",
        "🌾 Crop Recommendation",
        "📷 Soil Image Detection"
    ])

    with tab1:
        record_soil_tab()

    with tab2:
        view_soil_tab()

    with tab3:
        crop_recommendation_tab()

    with tab4:
        soil_image_detection_page()