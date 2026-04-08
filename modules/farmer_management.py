import streamlit as st
from database.db import (
    add_farmer, get_all_farmers,
    update_farmer, delete_farmer,
    get_crops, get_pest_logs
)


def farmer_management_page():
    st.title("👨‍🌾 Farmer Management")
    st.markdown("Register and manage farmer profiles linked to crops, soil records, and pest detections.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📝 Register Farmer", "👥 View Farmers"])

    # ── Register Tab ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Register New Farmer")

        name      = st.text_input("Full Name *", placeholder="e.g. Ravi Kumar")
        location  = st.text_input("Village / District / State", placeholder="e.g. Mysuru, Karnataka")
        land_area = st.number_input("Total Land (Acres)", min_value=0.0, step=0.5, format="%.2f")
        phone     = st.text_input("Contact Number", placeholder="e.g. 9876543210")

        if st.button("✅ Register Farmer", type="primary"):
            if not name.strip():
                st.error("❌ Full Name is required.")
            else:
                try:
                    add_farmer(
                        name=name.strip(),
                        phone=phone.strip(),
                        location=location.strip(),
                        land_area=land_area
                    )
                    st.success(f"✅ Farmer **{name}** registered successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Database error: {e}")

    # ── View Farmers Tab ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("All Registered Farmers")

        try:
            farmers = get_all_farmers()
        except Exception as e:
            st.error(f"Could not load farmers: {e}")
            return

        if not farmers:
            st.info("No farmers registered yet. Go to 'Register Farmer' tab to add one.")
            return

        st.write(f"**Total Farmers: {len(farmers)}**")
        st.markdown("---")

        for farmer in farmers:
            fid       = farmer["id"]
            fname     = farmer["name"]
            fphone    = farmer["phone"]    or "—"
            flocation = farmer["location"] or "—"
            fland     = float(farmer["land_area"]) if farmer["land_area"] else 0.0

            with st.expander(f"👨‍🌾 {fname}  |  📍 {flocation}  |  🌾 {fland} acres"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {fid}")
                    st.write(f"**Name:** {fname}")
                    st.write(f"**Phone:** {fphone}")
                with col2:
                    st.write(f"**Location:** {flocation}")
                    st.write(f"**Land Area:** {fland} acres")

                # Crop summary
                try:
                    crops = get_crops(fid)
                    if crops:
                        st.markdown(f"**🌱 Crops ({len(crops)}):**")
                        for crop in crops[:5]:
                            st.write(f"  • {crop['crop_name']} — {crop['status']} (planted {crop['planting_date']})")
                except Exception:
                    pass

                # Pest log summary
                try:
                    logs = get_pest_logs(fid)
                    if logs:
                        st.markdown(f"**🔍 Disease Detections ({len(logs)}):**")
                        for log in logs[:3]:
                            conf = f"{log['confidence']*100:.1f}%" if log['confidence'] else "—"
                            st.write(f"  • {log['disease_detected']} ({conf}) — {log['detected_at']}")
                except Exception:
                    pass

                st.markdown("---")
                ecol1, ecol2 = st.columns(2)

                with ecol1:
                    with st.popover("✏️ Edit Farmer"):
                        new_name     = st.text_input("Name",     value=fname,                                      key=f"en_{fid}")
                        new_location = st.text_input("Location", value=flocation if flocation != "—" else "",     key=f"el_{fid}")
                        new_land     = st.number_input("Land (Acres)", value=fland, min_value=0.0, step=0.5,      key=f"eld_{fid}")
                        new_phone    = st.text_input("Phone",    value=fphone    if fphone    != "—" else "",     key=f"ep_{fid}")
                        if st.button("💾 Save Changes", key=f"save_{fid}"):
                            try:
                                update_farmer(fid, new_name, new_phone, new_location, new_land)
                                st.success("Updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Update failed: {e}")

                with ecol2:
                    if st.button("🗑️ Delete", key=f"del_{fid}", type="secondary"):
                        try:
                            delete_farmer(fid)
                            st.success(f"Deleted farmer {fname}.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")