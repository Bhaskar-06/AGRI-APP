import streamlit as st
from database.db import add_crop, get_crops_by_farmer, get_all_farmers
import pandas as pd

def crop_management_page():
    st.title("🌱 Crop Management")
    farmers = get_all_farmers()

    if not farmers:
        st.warning("Please register a farmer first.")
        return

    farmer_map = {f"[{row[0]}] {row[1]}": row[0] for row in farmers}
    tab1, tab2 = st.tabs(["Add Crop Record", "View Crops"])

    with tab1:
        with st.form("crop_form"):
            selected = st.selectbox("Select Farmer", list(farmer_map.keys()))
            farmer_id = farmer_map[selected]
            crop_type = st.selectbox("Crop Type", [
                "Wheat", "Rice", "Maize", "Soybean", "Cotton",
                "Sugarcane", "Tomato", "Potato", "Onion", "Chickpea", "Other"
            ])
            field_name       = st.text_input("Field / Plot Name")
            planting_date    = st.date_input("Planting Date")
            expected_harvest = st.date_input("Expected Harvest Date")
            area_acres       = st.number_input("Field Area (Acres)", min_value=0.0, step=0.1)
            notes            = st.text_area("Additional Notes")
            submit = st.form_submit_button("Save Crop Record")

        if submit:
            add_crop(farmer_id, crop_type, field_name,
                     str(planting_date), str(expected_harvest), area_acres, notes)
            st.success("✅ Crop record saved!")

    with tab2:
        selected2 = st.selectbox("Filter by Farmer", list(farmer_map.keys()), key="view_sel")
        fid = farmer_map[selected2]
        crops = get_crops_by_farmer(fid)
        if crops:
            df = pd.DataFrame(crops, columns=[
                "ID","FarmerID","Crop","Field","Planting","Harvest","Area(ac)","Notes"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No crops found for this farmer.")