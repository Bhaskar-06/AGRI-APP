import streamlit as st

# ── MUST be first Streamlit command ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌾",
    layout="wide"
)

from database.db import init_db, get_connection
from modules.farmer_management import farmer_management_page
from modules.crop_management import crop_management_page
from modules.pest_detection import pest_detection_page
from modules.soil_health import soil_health_page

# Initialize DB
init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────
import os
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_column_width=True)

st.sidebar.title("🌾 Smart Agriculture")
st.sidebar.markdown("AI-Powered Farm Management")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "👨‍🌾 Farmer Management",
    "🌱 Crop Management",
    "🔍 Pest Detection",
    "🧪 Soil Health",
])

# ── Pages ─────────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🌾 AI Smart Agriculture Dashboard")
    st.markdown("""
    ### Welcome to your Smart Farm Management System
    Use the sidebar to navigate between modules:
    - **Farmer Management** — Register and manage farmer profiles
    - **Crop Management** — Track planting schedules and field data
    - **Pest Detection** — Upload leaf images for AI diagnosis
    - **Soil Health** — Get crop & fertilizer recommendations
    """)
    conn = get_connection()
    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Farmers", conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0])
    c2.metric("🌱 Crop Records", conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0])
    c3.metric("🧪 Soil Records", conn.execute("SELECT COUNT(*) FROM soil_records").fetchone()[0])
    conn.close()

elif page == "👨‍🌾 Farmer Management":
    farmer_management_page()

elif page == "🌱 Crop Management":
    crop_management_page()

elif page == "🔍 Pest Detection":
    pest_detection_page()

elif page == "🧪 Soil Health":
    soil_health_page()