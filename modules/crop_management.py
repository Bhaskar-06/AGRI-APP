import streamlit as st
import pandas as pd
from database.db import get_connection

def crop_management_page():
    st.title("🌱 Crop Management")

    try:
        conn = get_connection()
        farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if not farmers:
        st.warning("⚠️ No farmers found. Please register a farmer first.")
        return

    farmer_options = {f"{r['id']} - {r['name']}": r['id'] for r in farmers}
    tab1, tab2 = st.tabs(["Add Crop Record", "View Crop Records"])

    with tab1:
        st.subheader("Add New Crop Record")
        farmer_key   = st.selectbox("Select Farmer *", list(farmer_options.keys()), key="cm_farmer")
        crop_name    = st.text_input("Crop Name *", key="cm_crop")
        field_name   = st.text_input("Field Name", key="cm_field")
        area         = st.number_input("Area (Acres)", min_value=0.0, step=0.1, key="cm_area")
        plant_date   = st.date_input("Planting Date", key="cm_plant")
        harvest_date = st.date_input("Expected Harvest Date", key="cm_harvest")
        status       = st.selectbox("Status", ["Growing", "Harvested", "Failed", "Planned"], key="cm_status")
        notes        = st.text_area("Notes / Observations", key="cm_notes")

        if st.button("Save Crop Record", type="primary", key="cm_submit"):
            if not crop_name.strip():
                st.error("❌ Crop Name is required.")
            else:
                try:
                    fid = farmer_options[farmer_key]
                    conn = get_connection()
                    conn.execute(
                        """INSERT INTO crops (farmer_id, crop_name, field_name, area_acres,
                           planting_date, expected_harvest, status, notes)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (fid, crop_name.strip(), field_name.strip(), area,
                         str(plant_date), str(harvest_date), status, notes.strip())
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Crop **{crop_name}** saved successfully!")
                except Exception as e:
                    st.error(f"Database error: {e}")

    with tab2:
        st.subheader("All Crop Records")
        filter_farmer = st.selectbox("Filter by Farmer", ["All"] + list(farmer_options.keys()), key="cm_filter")
        try:
            conn = get_connection()
            if filter_farmer == "All":
                rows = conn.execute("""
                    SELECT c.id, f.name, c.crop_name, c.field_name, c.area_acres,
                           c.planting_date, c.expected_harvest, c.status, c.notes
                    FROM crops c JOIN farmers f ON c.farmer_id = f.id
                    ORDER BY c.created_at DESC
                """).fetchall()
            else:
                fid = farmer_options[filter_farmer]
                rows = conn.execute("""
                    SELECT c.id, f.name, c.crop_name, c.field_name, c.area_acres,
                           c.planting_date, c.expected_harvest, c.status, c.notes
                    FROM crops c JOIN farmers f ON c.farmer_id = f.id
                    WHERE c.farmer_id=? ORDER BY c.created_at DESC
                """, (fid,)).fetchall()
            conn.close()
            if rows:
                df = pd.DataFrame(rows, columns=["ID","Farmer","Crop","Field","Acres","Planted","Harvest","Status","Notes"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No crop records found.")
        except Exception as e:
            st.error(f"Database error: {e}")