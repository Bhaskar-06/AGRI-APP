import streamlit as st
import os

# ── MUST be first Streamlit command ──────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"   # ← Forces sidebar open on Cloud
)

from database.db import init_db, get_connection
from modules.farmer_management import farmer_management_page
from modules.crop_management import crop_management_page
from modules.pest_detection import pest_detection_page
from modules.soil_health import soil_health_page

# Initialize DB
init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 Smart Agriculture")
    st.markdown("**AI-Powered Farm Management**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=["🏠 Dashboard", "👨‍🌾 Farmer Management", "🌱 Crop Management", "🔍 Pest Detection", "🧪 Soil Health"],
        index=0
    )
    st.markdown("---")
    st.caption("© 2026 AI Smart Agriculture")

# ── Top Navigation Buttons (visible even if sidebar is collapsed) ─────────────
st.markdown("### 🌾 AI Smart Agriculture")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🏠 Dashboard", use_container_width=True):
        st.session_state["page"] = "🏠 Dashboard"
with col2:
    if st.button("👨‍🌾 Farmers", use_container_width=True):
        st.session_state["page"] = "👨‍🌾 Farmer Management"
with col3:
    if st.button("🌱 Crops", use_container_width=True):
        st.session_state["page"] = "🌱 Crop Management"
with col4:
    if st.button("🔍 Pest Detection", use_container_width=True):
        st.session_state["page"] = "🔍 Pest Detection"
with col5:
    if st.button("🧪 Soil Health", use_container_width=True):
        st.session_state["page"] = "🧪 Soil Health"

st.markdown("---")

# Use session_state button click OR sidebar radio (whichever was last used)
active_page = st.session_state.get("page", page)
# Sidebar radio always overrides if user clicks it
if page != st.session_state.get("_last_sidebar", page):
    active_page = page
st.session_state["_last_sidebar"] = page

# ── Pages ─────────────────────────────────────────────────────────────────────
if active_page == "🏠 Dashboard":
    st.title("🌾 AI Smart Agriculture Dashboard")
    st.markdown("""
    ### Welcome to your Smart Farm Management System
    Use the **sidebar** or **buttons above** to navigate between modules:
    - **👨‍🌾 Farmer Management** — Register and manage farmer profiles
    - **🌱 Crop Management** — Track planting schedules and field data
    - **🔍 Pest Detection** — Upload leaf images for AI diagnosis
    - **🧪 Soil Health** — Get crop & fertilizer recommendations
    """)
    conn = get_connection()
    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Farmers", conn.execute("SELECT COUNT(*) FROM farmers").fetchone()[0])
    c2.metric("🌱 Crop Records", conn.execute("SELECT COUNT(*) FROM crops").fetchone()[0])
    c3.metric("🧪 Soil Records", conn.execute("SELECT COUNT(*) FROM soil_records").fetchone()[0])
    conn.close()

elif active_page == "👨‍🌾 Farmer Management":
    farmer_management_page()

elif active_page == "🌱 Crop Management":
    crop_management_page()

elif active_page == "🔍 Pest Detection":
    pest_detection_page()

elif active_page == "🧪 Soil Health":
    soil_health_page()