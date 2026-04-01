import streamlit as st
import os

st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database.db import init_db, get_connection
from modules.farmer_management import farmer_management_page
from modules.crop_management import crop_management_page
from modules.soil_health import soil_health_page
from plant import pest_detection_page   # Combined pest + disease module

init_db()

# ── Session State ─────────────────────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Dashboard"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 Smart Agriculture")
    st.markdown("**AI-Powered Farm Management**")
    st.markdown("---")

    pages = [
        "🏠 Dashboard",
        "👨‍🌾 Farmer Management",
        "🌱 Crop Management",
        "🔍 Pest & Disease Detection",
        "🧪 Soil Health",
    ]

    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state["active_page"] = p

    st.markdown("---")
    st.caption("© 2026 AI Smart Agriculture")

# ── Top Nav ───────────────────────────────────────────────────────────────────
st.markdown("### 🌾 AI Smart Agriculture")

nav_labels = ["🏠 Dashboard", "👨‍🌾 Farmers", "🌱 Crops", "🔍 Pest & Disease", "🧪 Soil Health"]
nav_pages  = ["🏠 Dashboard", "👨‍🌾 Farmer Management", "🌱 Crop Management", "🔍 Pest & Disease Detection", "🧪 Soil Health"]

cols = st.columns(len(nav_labels))
for i, col in enumerate(cols):
    with col:
        if st.button(nav_labels[i], key=f"top_{i}", use_container_width=True):
            st.session_state["active_page"] = nav_pages[i]

st.markdown("---")

active_page = st.session_state["active_page"]

# ── Pages ─────────────────────────────────────────────────────────────────────
if active_page == "🏠 Dashboard":
    st.title("🌾 AI Smart Agriculture Dashboard")
    st.markdown("""
### Welcome to your Smart Farm Management System

Use the **sidebar** or **buttons above** to navigate:

- **👨‍🌾 Farmer Management** — Register and manage farmer profiles
- **🌱 Crop Management** — Track planting schedules + get pest protection advice
- **🔍 Pest & Disease Detection** — Upload leaf images for AI diagnosis of both pests and diseases
- **🧪 Soil Health** — Get crop & fertilizer recommendations
""")

    conn = get_connection()
    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Farmers",      conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0])
    c2.metric("🌱 Crop Records",   conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0])
    c3.metric("🧪 Soil Records",   conn.execute("SELECT COUNT(*) FROM soil_records").fetchone()[0])
    conn.close()

elif active_page == "👨‍🌾 Farmer Management":
    farmer_management_page()

elif active_page == "🌱 Crop Management":
    crop_management_page()

elif active_page == "🔍 Pest & Disease Detection":
    pest_detection_page()

elif active_page == "🧪 Soil Health":
    soil_health_page()