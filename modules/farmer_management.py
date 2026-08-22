import streamlit as st
import pandas as pd
from database.db import (
    db_add_farmer,
    db_get_farmers,
    db_delete_farmer,
    db_get_farmer_count
)

# ─────────────────────────────────────────
def register_farmer_tab():
    st.markdown("### 📋 Register New Farmer")

    with st.form("farmer_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name *", placeholder="e.g. Ravi Kumar")
            location = st.text_input("Village / District / State", placeholder="e.g. Mysuru, Karnataka")

        with col2:
            land_acres = st.number_input("Total Land (Acres)", min_value=0.0, step=0.5, format="%.2f")
            contact = st.text_input("Contact Number", placeholder="e.g. 9876543210", max_chars=10)

        submit = st.form_submit_button("✅ Register Farmer", type="primary", use_container_width=True)

        if submit:
            if not name.strip():
                st.error("❌ Full Name is required!")
                return
            if contact and not contact.isdigit():
                st.error("❌ Contact must be numbers only!")
                return

            success, msg = db_add_farmer(name.strip(), location, land_acres, contact)
            if success:
                st.success(f"✅ Farmer '{name}' registered successfully!")
                st.balloons()
            else:
                st.error(f"❌ Failed to save: {msg}")


# ─────────────────────────────────────────
def view_farmers_tab():
    st.markdown("### 👥 All Registered Farmers")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    farmers = db_get_farmers()

    if not farmers:
        st.warning("⚠️ No farmers registered yet.")
        st.info("Go to 'Register Farmer' tab to add farmers.")
        return

    total_land = sum(f.get("land_acres", 0) for f in farmers)
    avg_land = total_land / len(farmers) if farmers else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("👨‍🌾 Total Farmers", len(farmers))
    c2.metric("🌾 Total Land", f"{total_land:.1f} Acres")
    c3.metric("📊 Avg Land", f"{avg_land:.1f} Acres")

    st.markdown("---")

    rows = []
    for f in farmers:
        rows.append({
            "ID": f.get("id", ""),
            "Name": f.get("name", ""),
            "Location": f.get("location", ""),
            "Land(Acres)": f.get("land_acres", 0),
            "Contact": f.get("contact", ""),
            "Registered": str(f.get("created_at", ""))[:10]
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "farmers.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗑️ Remove a Farmer")

    name_to_id = {f["name"]: f["id"] for f in farmers}
    selected = st.selectbox("Select farmer to remove:", list(name_to_id.keys()))

    if st.button("❌ Delete Selected Farmer", type="secondary"):
        fid = name_to_id[selected]
        if db_delete_farmer(fid):
            st.success(f"✅ '{selected}' removed!")
            st.rerun()
        else:
            st.error("❌ Delete failed!")


# ─────────────────────────────────────────
def farmer_management_page():
    st.markdown("# 👨‍🌾 Farmer Management")
    st.markdown("Register and manage farmer profiles linked to crops, soil records, and pest detections.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📋 Register Farmer", "👥 View Farmers"])

    with tab1:
        register_farmer_tab()

    with tab2:
        view_farmers_tab()