import streamlit as st

st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database.db import (
    db_get_farmer_count,
    db_get_crop_count,
    db_get_soil_count
)
from modules.farmer_management import farmer_management_page
from modules.crop_management import crop_management_page
from modules.soil_health import soil_health_page
from plant import pest_detection_page

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Dashboard"

with st.sidebar:
    st.title("🌾 Smart Agriculture")
    st.markdown("AI-Powered Farm Management")
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

st.markdown("### 🌾 AI Smart Agriculture")

nav_labels = ["🏠 Dashboard", "👨‍🌾 Farmers", "🌱 Crops", "🔍 Pest & Disease", "🧪 Soil Health"]
nav_pages = [
    "🏠 Dashboard",
    "👨‍🌾 Farmer Management",
    "🌱 Crop Management",
    "🔍 Pest & Disease Detection",
    "🧪 Soil Health"
]

cols = st.columns(len(nav_labels))
for i, col in enumerate(cols):
    with col:
        if st.button(nav_labels[i], key=f"top_{i}", use_container_width=True):
            st.session_state["active_page"] = nav_pages[i]

st.markdown("---")

active_page = st.session_state["active_page"]

if active_page == "🏠 Dashboard":
    st.title("🌾 AI Smart Agriculture Dashboard")
    st.markdown("""
Welcome to your Smart Farm Management System.

Use the sidebar or the buttons above to navigate:

- 👨‍🌾 **Farmer Management** — Register and manage farmer profiles
- 🌱 **Crop Management** — Track planting schedules + get pest protection advice
- 🔍 **Pest & Disease Detection** — Upload leaf images for AI diagnosis
- 🧪 **Soil Health** — Log soil tests & get crop recommendations
    """)

    st.markdown("---")

    try:
        f_count = db_get_farmer_count()
        c_count = db_get_crop_count()
        s_count = db_get_soil_count()
    except Exception:
        f_count = c_count = s_count = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Farmers", f_count)
    c2.metric("🌱 Crop Records", c_count)
    c3.metric("🧪 Soil Records", s_count)

elif active_page == "👨‍🌾 Farmer Management":
    farmer_management_page()

elif active_page == "🌱 Crop Management":
    crop_management_page()

elif active_page == "🔍 Pest & Disease Detection":
    pest_detection_page()

elif active_page == "🧪 Soil Health":
    soil_health_page()