import streamlit as st
from database.db import (
    add_crop, get_crops, update_crop, delete_crop,
    get_all_farmers
)

CROP_LIST = [
    "Rice", "Wheat", "Maize / Corn", "Bajra / Pearl Millet", "Jowar / Sorghum",
    "Sugarcane", "Cotton", "Groundnut", "Soybean", "Sunflower",
    "Tomato", "Potato", "Onion", "Brinjal / Eggplant", "Chilli / Pepper",
    "Cabbage", "Cauliflower", "Spinach", "Okra / Bhindi", "Peas",
    "Mango", "Banana", "Grapes", "Pomegranate", "Apple",
    "Orange", "Watermelon", "Papaya", "Coconut", "Turmeric",
    "Ginger", "Garlic", "Mustard", "Lentil / Dal", "Chickpea / Gram",
    "Other"
]

STATUS_OPTIONS = ["Growing", "Harvested", "Diseased", "Pending Harvest", "Failed"]

STATUS_EMOJI = {
    "Growing":         "🌱",
    "Harvested":       "✅",
    "Diseased":        "🔴",
    "Pending Harvest": "🟡",
    "Failed":          "❌",
}

PEST_TIPS = {
    "Rice":             "Watch for Brown Plant Hopper and Blast disease. Use resistant varieties.",
    "Wheat":            "Monitor for Rust (yellow/brown/black). Apply propiconazole if spotted.",
    "Maize / Corn":     "Check for Fall Armyworm. Apply chlorpyrifos at early infestation.",
    "Tomato":           "Watch for Early/Late Blight and whitefly. Use copper fungicide.",
    "Potato":           "Late Blight is critical — apply Ridomil Gold at first symptoms.",
    "Cotton":           "Monitor for Bollworm. Use Bt-cotton varieties or spinosad spray.",
    "Sugarcane":        "Check for top borer and red rot. Use hot water seed treatment.",
    "Onion":            "Watch for purple blotch. Apply mancozeb every 10 days.",
    "Chilli / Pepper":  "Thrips and mites are common. Use abamectin or neem oil.",
    "Groundnut":        "Monitor for leaf spot and tikka disease. Apply carbendazim.",
    "Banana":           "Panama wilt and Sigatoka leaf spot — use proper drainage.",
    "Grapes":           "Downy mildew risk in humid conditions. Apply copper fungicide.",
}


def crop_management_page():
    st.title("🌱 Crop Management")
    st.markdown("Track your crops, planting schedules, harvest timelines, and get pest protection advice.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["➕ Add Crop Record", "📋 View All Crops", "🛡️ Pest Protection Tips"])

    # ── ADD CROP ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Add New Crop Record")

        # Farmer selection
        try:
            farmers = get_all_farmers()
        except Exception as e:
            st.error(f"Could not load farmers: {e}")
            return

        if not farmers:
            st.warning("⚠️ No farmers registered yet. Please register a farmer first.")
            return

        farmer_map = {f"[{f['id']}] {f['name']}": f["id"] for f in farmers}
        selected_farmer = st.selectbox("👨‍🌾 Select Farmer *", list(farmer_map.keys()))
        farmer_id = farmer_map[selected_farmer]

        col1, col2 = st.columns(2)
        with col1:
            crop_name = st.selectbox("🌾 Crop Name *", CROP_LIST)
            if crop_name == "Other":
                crop_name = st.text_input("Enter crop name", placeholder="e.g. Jowar")

            planting_date = st.date_input("📅 Planting Date")
            area = st.number_input("📐 Field Area (Acres)", min_value=0.0, step=0.5, format="%.2f")

        with col2:
            expected_harvest = st.date_input("🗓️ Expected Harvest Date")
            status = st.selectbox("📊 Status", STATUS_OPTIONS)
            notes = st.text_area("📝 Notes / Observations", placeholder="Any observations about this crop...", height=122)

        st.markdown("")
        if st.button("💾 Save Crop Record", type="primary"):
            if not crop_name or crop_name.strip() == "":
                st.error("❌ Crop name is required.")
            else:
                try:
                    add_crop(
                        farmer_id=farmer_id,
                        crop_name=crop_name.strip(),
                        planting_date=str(planting_date),
                        expected_harvest=str(expected_harvest),
                        area=area,
                        status=status,
                        notes=notes.strip()
                    )
                    st.success(f"✅ **{crop_name}** crop record saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Database error: {e}")

    # ── VIEW CROPS ────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("All Crop Records")

        # Filter by farmer
        try:
            farmers = get_all_farmers()
        except Exception:
            farmers = []

        filter_options = ["All Farmers"] + [f"[{f['id']}] {f['name']}" for f in farmers]
        filter_sel = st.selectbox("🔍 Filter by Farmer", filter_options, key="filter_farmer")

        filter_id = None
        if filter_sel != "All Farmers":
            filter_id = int(filter_sel.split("]")[0].replace("[", ""))

        try:
            crops = get_crops(filter_id)
        except Exception as e:
            st.error(f"Could not load crops: {e}")
            return

        if not crops:
            st.info("No crop records found. Add one from the 'Add Crop Record' tab.")
            return

        # Summary metrics
        total = len(crops)
        growing   = sum(1 for c in crops if c["status"] == "Growing")
        harvested = sum(1 for c in crops if c["status"] == "Harvested")
        diseased  = sum(1 for c in crops if c["status"] == "Diseased")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", total)
        m2.metric("🌱 Growing", growing)
        m3.metric("✅ Harvested", harvested)
        m4.metric("🔴 Diseased", diseased)
        st.markdown("---")

        # Build farmer id→name map for display
        farmer_name_map = {f["id"]: f["name"] for f in farmers}

        for crop in crops:
            cid       = crop["id"]
            cname     = crop["crop_name"]
            cstatus   = crop["status"] or "Growing"
            cplant    = crop["planting_date"] or "—"
            charvest  = crop["expected_harvest"] or "—"
            carea     = crop["area"] if crop["area"] else 0.0
            cnotes    = crop["notes"] or ""
            cfid      = crop["farmer_id"]
            cfarmer   = farmer_name_map.get(cfid, f"Farmer #{cfid}")
            emoji     = STATUS_EMOJI.get(cstatus, "🌿")

            with st.expander(f"{emoji} {cname}  |  👨‍🌾 {cfarmer}  |  📊 {cstatus}  |  🌾 {carea} acres"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Crop:** {cname}")
                    st.write(f"**Farmer:** {cfarmer}")
                    st.write(f"**Status:** {emoji} {cstatus}")
                with c2:
                    st.write(f"**Planted:** {cplant}")
                    st.write(f"**Harvest:** {charvest}")
                    st.write(f"**Area:** {carea} acres")

                if cnotes:
                    st.info(f"📝 **Notes:** {cnotes}")

                # Pest tip for this crop
                tip = PEST_TIPS.get(cname)
                if tip:
                    st.warning(f"🛡️ **Pest Tip:** {tip}")

                st.markdown("---")
                ecol1, ecol2 = st.columns(2)

                # Edit
                with ecol1:
                    with st.popover("✏️ Edit"):
                        new_name     = st.selectbox("Crop", CROP_LIST,
                                            index=CROP_LIST.index(cname) if cname in CROP_LIST else 0,
                                            key=f"cn_{cid}")
                        new_status   = st.selectbox("Status", STATUS_OPTIONS,
                                            index=STATUS_OPTIONS.index(cstatus) if cstatus in STATUS_OPTIONS else 0,
                                            key=f"cs_{cid}")
                        new_area     = st.number_input("Area (Acres)", value=float(carea),
                                            min_value=0.0, step=0.5, key=f"ca_{cid}")
                        new_notes    = st.text_area("Notes", value=cnotes, key=f"cn2_{cid}")
                        if st.button("💾 Save", key=f"save_{cid}"):
                            try:
                                update_crop(cid, new_name, cplant, charvest,
                                            new_area, new_status, new_notes)
                                st.success("Updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Update failed: {e}")

                # Delete
                with ecol2:
                    if st.button(f"🗑️ Delete", key=f"del_{cid}", type="secondary"):
                        try:
                            delete_crop(cid)
                            st.success(f"Deleted {cname} record.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

    # ── PEST TIPS ─────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("🛡️ Pest Protection Tips by Crop")
        st.markdown("Quick reference guide for common pests and diseases affecting your crops.")
        st.markdown("---")

        for crop, tip in PEST_TIPS.items():
            with st.expander(f"🌾 {crop}"):
                st.warning(f"⚠️ {tip}")

        st.markdown("---")
        st.info(
            "💡 **General Best Practices:**\n\n"
            "- Rotate crops every season to break pest cycles\n"
            "- Use certified disease-free seeds\n"
            "- Monitor fields every 3–5 days during critical growth stages\n"
            "- Apply pesticides early morning or evening to minimize bee impact\n"
            "- Keep field drainage clear to prevent fungal diseases"
        )