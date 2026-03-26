import streamlit as st
import pandas as pd
from database.db import get_connection

def farmer_management_page():
    st.title("👨‍🌾 Farmer Management")

    tab1, tab2 = st.tabs(["Register Farmer", "View Farmers"])

    with tab1:
        st.subheader("Register New Farmer")
        with st.form("farmer_form", clear_on_submit=True):
            name     = st.text_input("Full Name *")
            location = st.text_input("Village / District / State")
            land     = st.number_input("Total Land (Acres)", min_value=0.0, step=0.5)
            contact  = st.text_input("Contact Number")
            submitted = st.form_submit_button("Register Farmer")

        if submitted:
            if not name.strip():
                st.error("Full Name is required.")
            else:
                conn = get_connection()
                conn.execute(
                    "INSERT INTO farmers (name, location, land_acres, contact) VALUES (?,?,?,?)",
                    (name.strip(), location.strip(), land, contact.strip())
                )
                conn.commit()
                conn.close()
                st.success(f"✅ Farmer **{name}** registered successfully!")

    with tab2:
        st.subheader("All Registered Farmers")
        conn = get_connection()
        rows = conn.execute("SELECT * FROM farmers ORDER BY created_at DESC").fetchall()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=["ID", "Name", "Location", "Land (Acres)", "Contact", "Registered On"])
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader("🗑️ Delete Farmer")
            farmer_ids = [f"{r['id']} - {r['name']}" for r in rows]
            selected = st.selectbox("Select Farmer to Delete", farmer_ids)
            if st.button("Delete Farmer", type="primary"):
                fid = int(selected.split(" - ")[0])
                conn = get_connection()
                conn.execute("DELETE FROM farmers WHERE id=?", (fid,))
                conn.execute("DELETE FROM crops WHERE farmer_id=?", (fid,))
                conn.execute("DELETE FROM soil_records WHERE farmer_id=?", (fid,))
                conn.commit()
                conn.close()
                st.success("Farmer and all related records deleted.")
                st.rerun()
        else:
            st.info("No farmers registered yet. Use the 'Register Farmer' tab to add one.")