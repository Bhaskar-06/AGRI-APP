import os
import pickle
import numpy as np
import pandas as pd
import requests
import streamlit as st
from database.db import db_add_soil, db_get_soil_records, db_get_farmers
from modules.soil_image_detection import soil_image_detection_page

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

# Maps a farmer-friendly category to a representative numeric value the
# ML model expects — same categories used on India's government-issued
# Soil Health Card, so most farmers already recognize this language.
NPK_LEVELS = {"Low": 20, "Medium": 60, "High": 100}
P_LEVELS = {"Low": 15, "Medium": 40, "High": 80}
K_LEVELS = {"Low": 15, "Medium": 40, "High": 80}
PH_LEVELS = {"Acidic (sour soil, e.g. red/laterite areas)": 5.5,
             "Neutral (most farmland)": 6.5,
             "Alkaline (whitish/salty soil)": 8.0}
MOISTURE_LEVELS = {"Dry": 20, "Normal": 50, "Wet": 80}


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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_for_place(place_name):
    """Auto-fills temperature, humidity, and rainfall from a village/town
    name so the farmer never has to know these numbers themselves.
    Uses Open-Meteo — free, no API key required."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1}, timeout=10
        ).json()
        if not geo.get("results"):
            return None
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum",
                "past_days": 30, "forecast_days": 1,
                "timezone": "auto",
            }, timeout=10
        ).json()

        temp = weather["current"]["temperature_2m"]
        humidity = weather["current"]["relative_humidity_2m"]
        recent_rain_30d = sum(weather["daily"]["precipitation_sum"])
        estimated_annual_rainfall = recent_rain_30d * 12  # rough yearly estimate

        return {"temperature": temp, "humidity": humidity, "rainfall": estimated_annual_rainfall}
    except Exception:
        return None


def _simple_soil_inputs():
    """Farmer-friendly inputs: no lab numbers, no jargon."""
    st.markdown("#### 📍 Where is your field?")
    place = st.text_input(
        "Village / Town name",
        placeholder="e.g. Mysuru",
        help="We'll automatically fetch temperature, humidity and rainfall for this location."
    )

    weather = None
    if place:
        with st.spinner("Fetching local weather data..."):
            weather = fetch_weather_for_place(place)
        if weather:
            st.success(
                f"✅ Auto-filled from {place}: "
                f"{weather['temperature']:.1f}°C, {weather['humidity']:.0f}% humidity, "
                f"~{weather['rainfall']:.0f}mm/yr rainfall estimate"
            )
        else:
            st.warning("⚠️ Couldn't find that location — using typical average values instead.")

    st.markdown("#### 🌱 About your soil")
    st.caption("If you have a Soil Health Card from the government, use those N/P/K ratings here.")

    col1, col2 = st.columns(2)
    with col1:
        n_choice = st.selectbox("Nitrogen level", list(NPK_LEVELS.keys()), index=1, key="simple_n")
        p_choice = st.selectbox("Phosphorus level", list(P_LEVELS.keys()), index=1, key="simple_p")
        k_choice = st.selectbox("Potassium level", list(K_LEVELS.keys()), index=1, key="simple_k")
    with col2:
        ph_choice = st.selectbox("Soil type", list(PH_LEVELS.keys()), index=1, key="simple_ph")
        moisture_choice = st.selectbox("How wet is the soil right now?", list(MOISTURE_LEVELS.keys()), index=1, key="simple_moi")

    return {
        "N": NPK_LEVELS[n_choice], "P": P_LEVELS[p_choice], "K": K_LEVELS[k_choice],
        "ph": PH_LEVELS[ph_choice], "moisture": MOISTURE_LEVELS[moisture_choice],
        "temp": weather["temperature"] if weather else 27.0,
        "hum": weather["humidity"] if weather else 65.0,
        "rain": weather["rainfall"] if weather else 500.0,
    }


def _advanced_soil_inputs():
    """For users with exact lab/sensor readings."""
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N) kg/ha", 0.0, 200.0, 50.0, key="adv_n")
        P = st.number_input("Phosphorus (P) kg/ha", 0.0, 200.0, 30.0, key="adv_p")
        K = st.number_input("Potassium (K) kg/ha", 0.0, 200.0, 40.0, key="adv_k")
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1, key="adv_ph")
    with col2:
        temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="adv_temp")
        hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="adv_hum")
        moi = st.number_input("Moisture (%)", 0.0, 100.0, 40.0, key="adv_moi")
        rain = st.number_input("Rainfall (mm/yr)", 0.0, 3000.0, 500.0, key="adv_rain")

    return {"N": N, "P": P, "K": K, "ph": ph, "moisture": moi, "temp": temp, "hum": hum, "rain": rain}


def record_soil_tab():
    st.subheader("Enter Soil Parameters")

    farmers = db_get_farmers()
    farmer_options = {"None (Guest)": None}
    farmer_options.update({f"{f['id']} - {f['name']}": f['id'] for f in farmers})

    if "soil_result" not in st.session_state:
        st.session_state["soil_result"] = None

    farmer_key = st.selectbox("Link to Farmer (optional)", list(farmer_options.keys()), key="sh_farmer")
    field_name = st.text_input("Field Name", key="sh_field")

    advanced = st.checkbox("📋 I have exact soil test numbers (advanced)", key="sh_advanced_toggle")
    st.markdown("---")

    values = _advanced_soil_inputs() if advanced else _simple_soil_inputs()

    if st.button("🔍 Get Recommendation", type="primary", key="sh_submit", use_container_width=True):
        with st.spinner("Analyzing soil data..."):
            try:
                model = load_recommender()
                features = np.array([[values["N"], values["P"], values["K"],
                                       values["temp"], values["hum"], values["ph"], values["rain"]]])
                recommended_crop = str(model.predict(features)[0])

                score, tips = 0, []
                if 6.0 <= values["ph"] <= 7.5:
                    score += 30
                else:
                    tips.append("⚠️ Soil pH is outside the ideal 6.0–7.5 range — consider lime (acidic) or gypsum (alkaline)")
                if values["N"] >= 40:
                    score += 20
                else:
                    tips.append("⚠️ Low Nitrogen — apply Urea or compost")
                if values["P"] >= 20:
                    score += 20
                else:
                    tips.append("⚠️ Low Phosphorus — apply DAP fertilizer")
                if values["K"] >= 20:
                    score += 20
                else:
                    tips.append("⚠️ Low Potassium — apply MOP (Muriate of Potash)")
                if values["hum"] >= 40:
                    score += 10
                else:
                    tips.append("⚠️ Low humidity — increase irrigation frequency")

                st.session_state["soil_result"] = {
                    "recommended_crop": recommended_crop, "score": score, "tips": tips,
                }

                db_add_soil(
                    farmer_id=farmer_options[farmer_key], field_name=field_name,
                    nitrogen=values["N"], phosphorus=values["P"], potassium=values["K"],
                    ph=values["ph"], temperature=values["temp"], humidity=values["hum"],
                    moisture=values["moisture"], rainfall=values["rain"],
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
        "N": r.get("nitrogen", ""), "P": r.get("phosphorus", ""), "K": r.get("potassium", ""),
        "pH": r.get("ph_level", ""), "Temp": r.get("temperature", ""), "Humidity": r.get("humidity", ""),
        "Rainfall": r.get("rainfall", ""), "Recommended Crop": r.get("recommended_crop", ""),
        "Date": str(r.get("created_at", ""))[:10],
    } for r in rows])

    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "soil_history.csv", "text/csv", use_container_width=True)


def crop_recommendation_tab():
    st.subheader("🌾 Quick Crop Recommendation")
    st.caption("No record saved — just a quick check.")

    model = load_recommender()
    advanced = st.checkbox("📋 I have exact soil test numbers (advanced)", key="cr_advanced_toggle")

    values = _advanced_soil_inputs() if advanced else _simple_soil_inputs()

    if st.button("🌾 Recommend Crop", type="primary", use_container_width=True, key="cr_btn"):
        try:
            features = np.array([[values["N"], values["P"], values["K"],
                                   values["temp"], values["hum"], values["ph"], values["rain"]]])
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
        "➕ Record Soil Data", "📋 View Records", "🌾 Crop Recommendation", "📷 Soil Image Detection"
    ])

    with tab1:
        record_soil_tab()
    with tab2:
        view_soil_tab()
    with tab3:
        crop_recommendation_tab()
    with tab4:
        soil_image_detection_page()