import streamlit as st
import pandas as pd
from database.db import get_connection

def farmer_management_page():
    st.title("👨‍🌾 Farmer Management")
    tab1, tab2 = st.tabs(["Register Farmer", "View Farmers"])

    with tab1:
        st.subheader("Register New Farmer")
        name     = st.text_input("Full Name *", key="fm_name")
        location = st.text_input("Village / District / State", key="fm_loc")
        land     = st.number_input("Total Land (Acres)", min_value=0.0, step=0.5, key="fm_land")
        contact  = st.text_input("Contact Number", key="fm_contact")

        if st.button("Register Farmer", type="primary", key="fm_submit"):
            if not name.strip():
                st.error("❌ Full Name is required.")
            else:
                try:
                    conn = get_connection()
                    conn.execute(
                        "INSERT INTO farmers (name, location, land_acres, contact) VALUES (?,?,?,?)",
                        (name.strip(), location.strip(), land, contact.strip())
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Farmer **{name}** registered successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Database error: {e}")

    with tab2:
        st.subheader("All Registered Farmers")
        try:
            conn = get_connection()
            rows = conn.execute("SELECT id, name, location, land_acres, contact, created_at FROM farmers ORDER BY created_at DESC").fetchall()
            conn.close()

            if rows:
                df = pd.DataFrame(rows, columns=["ID", "Name", "Location", "Land (Acres)", "Contact", "Registered On"])
                st.dataframe(df, use_container_width=True)

                st.markdown("---")
                st.subheader("🗑️ Delete Farmer")
                farmer_list = [f"{r['id']} - {r['name']}" for r in rows]
                selected = st.selectbox("Select Farmer to Delete", farmer_list)
                if st.button("Delete Farmer", key="fm_delete"):
                    fid = int(selected.split(" - ")[0])
                    conn = get_connection()
                    conn.execute("DELETE FROM farmers WHERE id=?", (fid,))
                    conn.execute("DELETE FROM crops WHERE farmer_id=?", (fid,))
                    conn.execute("DELETE FROM soil_records WHERE farmer_id=?", (fid,))
                    conn.execute("DELETE FROM pest_detections WHERE farmer_id=?", (fid,))
                    conn.commit()
                    conn.close()
                    st.success("Farmer and all related records deleted.")
                    st.rerun()
            else:
                st.info("No farmers registered yet.")
        except Exception as e:
            st.error(f"Database error: {e}")