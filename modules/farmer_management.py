import streamlit as st
from database.db import add_farmer, get_all_farmers, get_crops_by_farmer
import pandas as pd

def farmer_management_page():
    st.title("👨‍🌾 Farmer Management")
    tab1, tab2 = st.tabs(["Register Farmer", "View Farmers"])

    with tab1:
        st.subheader("Register New Farmer")
        with st.form("farmer_form"):
            name     = st.text_input("Full Name")
            location = st.text_input("Village / District / State")
            acreage  = st.number_input("Total Land (Acres)", min_value=0.0, step=0.5)
            contact  = st.text_input("Contact Number")
            submit   = st.form_submit_button("Register Farmer")

        if submit:
            if name and contact:
                add_farmer(name, location, acreage, contact)
                st.success(f"✅ Farmer '{name}' registered successfully!")
            else:
                st.error("Name and Contact are required.")

    with tab2:
        st.subheader("All Registered Farmers")
        farmers = get_all_farmers()
        if farmers:
            df = pd.DataFrame(farmers, columns=["ID","Name","Location","Acreage","Contact","Registered"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No farmers registered yet.")