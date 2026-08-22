import streamlit as st
import pandas as pd
from datetime import date, timedelta
from database.db import (
    db_add_crop,
    db_get_crops,
    db_update_crop,
    db_delete_crop,
    db_get_farmers
)

# ─────────────────────────────────────────
# PEST PROTECTION TIPS DATABASE
# ─────────────────────────────────────────
PEST_TIPS = {
    "Rice": [
        "🌾 Apply Carbofuran 3G @ 25 kg/ha for stem borer control",
        "🔍 Monitor weekly for Brown Plant Hopper (BPH)",
        "💧 Maintain proper water level (5 cm) to reduce pests",
        "🌿 Use neem oil spray (5ml/L) for leaf folder control",
        "🪤 Install light traps @ 1 per acre for moth monitoring"
    ],
    "Wheat": [
        "🌡️ Monitor for Rust diseases during humid conditions",
        "🐛 Watch for Aphid colonies on leaves and stems",
        "💊 Apply Mancozeb 75% WP @ 2.5g/L for rust control",
        "🌿 Spray neem-based pesticide for aphid management",
        "🔄 Rotate with legumes to break pest cycles"
    ],
    "Tomato": [
        "🐛 Install yellow sticky traps for whitefly monitoring",
        "🍅 Apply Spinosad for fruit borer at fruit set stage",
        "🌿 Neem oil (5ml/L) spray every 7 days for mites",
        "🔍 Check undersides of leaves for early mite detection",
        "💧 Avoid overhead irrigation to reduce disease spread"
    ],
    "Cotton": [
        "🐛 Monitor Bollworm with pheromone traps",
        "🌿 Apply Bt spray for Bollworm early instar larvae",
        "🔍 Check for Jassid/Aphid infestation weekly",
        "💊 Imidacloprid 17.8% SL @ 0.5ml/L for sucking pests",
        "🪤 Install 5 pheromone traps per acre"
    ],
    "Maize": [
        "🐛 Watch for Fall Army Worm (FAW) - new major threat",
        "🌿 Apply Emamectin benzoate for FAW control",
        "🔍 Check whorl for FAW egg masses and feeding damage",
        "💊 Chlorpyrifos 20% EC @ 2.5ml/L for stem borer",
        "🪤 Use light traps and pheromone traps for monitoring"
    ],
    "Potato": [
        "🥔 Late blight is the biggest threat - monitor closely",
        "🌿 Apply Mancozeb + Metalaxyl for late blight control",
        "🐛 Watch for Colorado Beetle and aphid vectors",
        "💧 Avoid excess moisture to prevent tuber rot",
        "🔄 Use certified disease-free seed tubers"
    ],
    "Onion": [
        "🧅 Thrips is major pest - use blue sticky traps",
        "🌿 Spinosad 45% SC @ 0.3ml/L for thrips control",
        "💊 Propiconazole 25% EC for purple blotch disease",
        "💧 Drip irrigation to reduce leaf wetness",
        "🔍 Monitor for Thrips tabaci weekly"
    ],
    "Default": [
        "🔍 Monitor crop weekly for early pest detection",
        "🌿 Use neem oil (5ml/L) as general bio-pesticide",
        "📋 Maintain field diary for pest observations",
        "🔄 Practice crop rotation every season",
        "💧 Avoid excessive irrigation - promotes fungal diseases",
        "🌱 Use certified seeds for better disease resistance",
        "📞 Contact local agriculture officer for guidance"
    ]
}

# ─────────────────────────────────────────
# ADD CROP TAB
# ─────────────────────────────────────────
def add_crop_tab():
    st.markdown("### ➕ Add New Crop Record")

    farmers = db_get_farmers()

    if not farmers:
        st.warning("⚠️ No farmers registered yet. Please register a farmer first.")
        return

    farmer_options = {f["name"]: f["id"] for f in farmers}

    with st.form("add_crop_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            selected_farmer = st.selectbox("Select Farmer *", options=list(farmer_options.keys()))
            crop_name = st.selectbox(
                "Crop Name *",
                options=["Rice", "Wheat", "Tomato", "Cotton", "Maize", "Potato", "Onion",
                         "Sugarcane", "Soybean", "Groundnut", "Sunflower", "Chilli",
                         "Brinjal", "Cabbage", "Other"]
            )
            area_acres = st.number_input("Area (Acres)", min_value=0.1, step=0.5, format="%.2f")

        with col2:
            planting_date = st.date_input("Planting Date", value=date.today())
            harvest_date = st.date_input("Expected Harvest Date", value=date.today() + timedelta(days=90))
            status = st.selectbox("Status", options=["Active", "Harvested", "Failed", "Planned"])

        notes = st.text_area("Notes (Optional)", placeholder="Any special observations or notes...", height=80)

        submit = st.form_submit_button("✅ Add Crop Record", type="primary", use_container_width=True)

        if submit:
            if not crop_name:
                st.error("❌ Crop name is required!")
                return

            if harvest_date <= planting_date:
                st.error("❌ Harvest date must be after planting date!")
                return

            farmer_id = farmer_options[selected_farmer]
            success = db_add_crop(
                farmer_id=farmer_id,
                crop_name=crop_name,
                planting_date=planting_date,
                harvest_date=harvest_date,
                area_acres=area_acres,
                status=status,
                notes=notes
            )

            if success:
                st.success(f"✅ {crop_name} record added for {selected_farmer}!")
                st.balloons()
            else:
                st.error("❌ Failed to add crop record. Try again.")

# ─────────────────────────────────────────
# VIEW ALL CROPS TAB
# ─────────────────────────────────────────
def view_crops_tab():
    st.markdown("### 🌱 All Crop Records")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    crops = db_get_crops()

    if not crops:
        st.warning("⚠️ No crop records found.")
        st.info("Go to 'Add Crop Record' tab to add crops.")
        return

    total_crops = len(crops)
    active_crops = sum(1 for c in crops if c.get("status") == "Active")
    total_area = sum(c.get("area_acres", 0) for c in crops)

    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Total Crops", total_crops)
    c2.metric("✅ Active Crops", active_crops)
    c3.metric("🌾 Total Area", f"{total_area:.1f} Acres")

    st.markdown("---")

    rows = []
    for c in crops:
        farmer_info = c.get("farmers", {})
        farmer_name = farmer_info.get("name", "Unknown") if farmer_info else "Unknown"

        rows.append({
            "ID": c.get("id", ""),
            "Farmer": farmer_name,
            "Crop": c.get("crop_name", ""),
            "Area(Acres)": c.get("area_acres", 0),
            "Planted": str(c.get("planting_date", ""))[:10],
            "Harvest": str(c.get("harvest_date", ""))[:10],
            "Status": c.get("status", ""),
            "Notes": c.get("notes", "")
        })

    df = pd.DataFrame(rows)

    def color_status(val):
        colors = {
            "Active": "background-color: #1a472a; color: #4ade80",
            "Harvested": "background-color: #1e3a5f; color: #60a5fa",
            "Failed": "background-color: #4a1a1a; color: #f87171",
            "Planned": "background-color: #3a3a1a; color: #fbbf24"
        }
        return colors.get(val, "")

    styled_df = df.style.map(color_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "crop_records.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗑️ Delete Crop Record")

    crop_options = {
        f"#{c.get('id')} - {c.get('crop_name')} ({c.get('status')})": c.get("id")
        for c in crops
    }

    selected_crop = st.selectbox("Select crop to delete:", list(crop_options.keys()))

    if st.button("❌ Delete Selected Crop", type="secondary"):
        crop_id = crop_options[selected_crop]
        if db_delete_crop(crop_id):
            st.success("✅ Crop record deleted!")
            st.rerun()
        else:
            st.error("❌ Delete failed!")

# ─────────────────────────────────────────
# PEST PROTECTION TIPS TAB
# ─────────────────────────────────────────
def pest_tips_tab():
    st.markdown("### 🛡️ Pest Protection Tips")

    crop_list = ["Rice", "Wheat", "Tomato", "Cotton", "Maize", "Potato", "Onion", "Default"]
    selected = st.selectbox("Select crop for specific tips:", crop_list, index=0)

    tips = PEST_TIPS.get(selected, PEST_TIPS["Default"])

    st.markdown(f"#### Pest Protection Tips for {selected}")
    st.markdown("---")

    for i, tip in enumerate(tips, 1):
        st.markdown(f"**{i}.** {tip}")

    st.markdown("---")
    st.info("""
    📞 **Need Expert Help?**
    - Kisan Call Centre: **1800-180-1551** (Toll Free)
    - Crop Insurance: **1800-200-7710**
    - Local Agriculture Office: Visit your nearest Krishi Vigyan Kendra
    """)

# ─────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────
def crop_management_page():
    st.markdown("# 🌱 Crop Management")
    st.markdown("Track your crops, planting schedules, harvest timelines, and get pest protection advice.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["➕ Add Crop Record", "📋 View All Crops", "🛡️ Pest Protection Tips"])

    with tab1:
        add_crop_tab()

    with tab2:
        view_crops_tab()

    with tab3:
        pest_tips_tab()