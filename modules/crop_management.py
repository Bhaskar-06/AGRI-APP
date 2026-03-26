import streamlit as st
import pandas as pd
from database.db import get_connection

# ── Comprehensive pest protection advice for ALL major crop types
CROP_PEST_ADVICE = {
    "rice":       {"common_pests":"Brown planthopper, Stem borer, Leaf folder, Blast fungus, Bacterial leaf blight","chemical":"Chlorpyrifos 20EC @ 2ml/L for insects. Tricyclazole 75WP @ 0.6g/L for blast.","organic":"Neem oil 5ml/L + TNAU pheromone traps. Trichogramma egg parasitoid release.","cultural":"Use resistant varieties. Maintain proper water level. Remove weeds.","schedule":"Scout every 7 days. Spray at Economic Threshold Level (ETL)."},
    "wheat":      {"common_pests":"Aphids, Brown rust, Yellow rust, Karnal bunt, Loose smut","chemical":"Imidacloprid 70WG @ 0.3g/L for aphids. Propiconazole 25EC @ 1ml/L for rust.","organic":"Neem seed kernel extract 5% for aphids. Trichoderma seed treatment.","cultural":"Use certified seeds. Timely sowing. Avoid excess nitrogen.","schedule":"Monitor from tillering. Apply fungicide at flag leaf stage."},
    "maize":      {"common_pests":"Fall armyworm, Stem borer, Grey leaf spot, Northern leaf blight","chemical":"Emamectin benzoate 5SG @ 0.4g/L for armyworm. Azoxystrobin for leaf blight.","organic":"Bacillus thuringiensis (Bt) spray for armyworm. NPV biopesticide.","cultural":"Early planting. Intercrop with beans. Remove and destroy infested plants.","schedule":"Scout weekly from seedling stage. Apply Bt at whorl stage for armyworm."},
    "corn":       {"common_pests":"Fall armyworm, Corn earworm, Corn borer, Grey leaf spot","chemical":"Emamectin benzoate 5SG @ 0.4g/L or Chlorantraniliprole 18.5SC @ 0.3ml/L.","organic":"Bacillus thuringiensis spray. Trichogramma card release @ 1 lakh/ha.","cultural":"Crop rotation. Proper spacing 60x20cm. Remove crop debris after harvest.","schedule":"Scout from V3 stage. Economic threshold: 1 armyworm per plant."},
    "tomato":     {"common_pests":"Whitefly (TYLCV vector), Tomato borer, Early blight, Late blight, Spider mites","chemical":"Imidacloprid for whitefly. Chlorantraniliprole for borer. Metalaxyl+Mancozeb for late blight.","organic":"Yellow sticky traps. Neem oil 5ml/L. Trichoderma for soil-borne diseases.","cultural":"Stake all plants. Drip irrigation only. Remove infected leaves. Crop rotation.","schedule":"Scout twice weekly. Apply fungicide preventively in humid weather."},
    "potato":     {"common_pests":"Late blight, Early blight, Colorado potato beetle, Aphids, Potato cyst nematode","chemical":"Metalaxyl+Mancozeb for late blight. Chlorpyrifos for beetles. Carbofuran for nematodes.","organic":"Copper hydroxide spray. Spinosad for beetles. Neem cake in soil.","cultural":"Hill soil at 20cm. Use certified seed tubers. Destroy all volunteer plants.","schedule":"Monitor daily during cool wet weather for late blight (highest priority)."},
    "onion":      {"common_pests":"Thrips, Purple blotch, Stemphylium blight, Basal rot","chemical":"Fipronil 5SC @ 1.5ml/L for thrips. Iprodione 50WP @ 2g/L for purple blotch.","organic":"Neem oil 5ml/L for thrips. Trichoderma soil application.","cultural":"Proper spacing 15x10cm. Avoid excess nitrogen. Good field drainage.","schedule":"Scout weekly. Apply fungicide at bulb initiation stage."},
    "cotton":     {"common_pests":"Bollworm, Whitefly, Aphids, Jassids, Pink bollworm","chemical":"Spinosad 45SC @ 0.3ml/L for bollworm. Acetamiprid 20SP @ 0.3g/L for whitefly.","organic":"Bacillus thuringiensis spray. NPV for bollworm. Yellow sticky traps.","cultural":"Use Bt cotton varieties. Timely sowing. Install pheromone traps @ 5/ha.","schedule":"Scout twice weekly during boll development. Check 20 plants per row."},
    "soybean":    {"common_pests":"Girdle beetle, Stem fly, Leaf folder, Bacterial pustule, Rust","chemical":"Chlorantraniliprole 18.5SC for girdle beetle. Tebuconazole for rust.","organic":"Neem oil spray. Beauveria bassiana biopesticide for insects.","cultural":"Proper spacing 45x5cm. Crop rotation with cereals. Remove weeds early.","schedule":"Scout weekly from emergence. Critical period: flowering to pod fill."},
    "sugarcane":  {"common_pests":"Pyrilla (planthopper), Top borer, Red rot, Smut, Ratoon stunting disease","chemical":"Chlorpyrifos 20EC for pyrilla. Carbendazim 50WP for red rot.","organic":"Epipyrops melanoleuca egg parasitoid. Trichoderma for soil treatment.","cultural":"Hot water treatment of setts at 50°C for 2 hours. Remove ratoon stubble.","schedule":"Scout monthly. Trash mulching controls pests and conserves moisture."},
    "groundnut":  {"common_pests":"Leaf miner, Tikka disease (early/late leaf spot), Crown rot, Termites","chemical":"Chlorpyrifos for termites @ soil treatment. Chlorothalonil for leaf spots.","organic":"Neem oil spray. Trichoderma seed treatment at 4g/kg seed.","cultural":"Proper spacing 30x10cm. Avoid waterlogging. Crop rotation with cereals.","schedule":"Scout weekly from 30 days after sowing. Monitor for aflatoxin risk at harvest."},
    "mango":      {"common_pests":"Mango hopper, Fruit fly, Powdery mildew, Anthracnose, Malformation","chemical":"Imidacloprid for hopper. Fenthion for fruit fly. Carbendazim for anthracnose.","organic":"Kaolin clay spray for fruit fly. Neem oil for hopper. Copper for anthracnose.","cultural":"Prune for open canopy. Remove mummified fruits. Bag fruits for fruit fly.","schedule":"Spray at panicle emergence and pea stage for hopper and powdery mildew."},
    "banana":     {"common_pests":"Panama wilt (Fusarium), Sigatoka leaf spot, Banana weevil, Nematodes","chemical":"Propiconazole for Sigatoka. Chlorpyrifos for weevil. Carbofuran for nematodes.","organic":"Trichoderma application in planting pit. Pseudomonas fluorescens.","cultural":"Use disease-free suckers/TC plants. Remove affected leaves. Proper spacing.","schedule":"Monitor every 2 weeks. Record Sigatoka leaf spot index on 3rd and 5th leaves."},
    "grapes":     {"common_pests":"Downy mildew, Powdery mildew, Botrytis bunch rot, Grape berry moth, Mealy bugs","chemical":"Metalaxyl+Mancozeb for downy mildew. Tebuconazole for powdery mildew.","organic":"Bordeaux mixture 1% for downy mildew. Sulfur spray for powdery mildew.","cultural":"Canopy management for airflow. Bag bunches. Remove infected material.","schedule":"Spray protective fungicide from bud break. Critical at flowering and bunch closure."},
    "chilli":     {"common_pests":"Thrips, Mites, Aphids, Fruit borer, Anthracnose, Phytophthora blight","chemical":"Spinosad for thrips and borer. Abamectin for mites. Metalaxyl for blight.","organic":"Neem oil 5ml/L for thrips and mites. Yellow/blue sticky traps.","cultural":"Drip irrigation. Stake plants. Mulch with black polythene. Crop rotation.","schedule":"Scout twice weekly during flowering and fruit development."},
    "default":    {"common_pests":"Various insects, fungal pathogens, bacterial diseases","chemical":"Consult local agricultural extension officer for specific recommendations based on crop and pest.","organic":"Neem oil 5ml/L as broad-spectrum preventive. Trichoderma for soil health.","cultural":"Proper spacing, crop rotation, field hygiene, and balanced fertilization are essential.","schedule":"Scout crops at least once a week. Keep detailed records of observations."},
}

def get_crop_advice(crop_name: str) -> dict:
    crop_lower = crop_name.lower().strip()
    for key, advice in CROP_PEST_ADVICE.items():
        if key in crop_lower or crop_lower in key:
            return advice
    return CROP_PEST_ADVICE["default"]

def crop_management_page():
    st.title("🌱 Crop Management")

    conn    = get_connection()
    farmers = conn.execute("SELECT id, name FROM farmers").fetchall()
    conn.close()

    if not farmers:
        st.warning("⚠️ No farmers registered yet. Go to 👨‍🌾 Farmer Management first.")
        return

    farmer_options = {f"{r['id']} - {r['name']}": r['id'] for r in farmers}
    tab1, tab2, tab3 = st.tabs(["➕ Add Crop Record", "📋 View Records", "🛡️ Pest Protection Advice"])

    with tab1:
        st.subheader("Add New Crop Record")
        farmer_key   = st.selectbox("Select Farmer *", list(farmer_options.keys()), key="cm_farmer")
        crop_name    = st.text_input("Crop Name *", key="cm_crop", placeholder="e.g. Tomato, Rice, Wheat...")
        field_name   = st.text_input("Field Name", key="cm_field")
        area         = st.number_input("Area (Acres)", 0.0, step=0.1, key="cm_area")
        plant_date   = st.date_input("Planting Date", key="cm_plant")
        harvest_date = st.date_input("Expected Harvest Date", key="cm_harvest")
        status       = st.selectbox("Status", ["Growing","Harvested","Failed","Planned"], key="cm_status")
        notes        = st.text_area("Notes / Observations", key="cm_notes")

        if st.button("💾 Save Crop Record", type="primary", key="cm_submit"):
            if not crop_name.strip():
                st.error("❌ Crop Name is required.")
            else:
                try:
                    fid = farmer_options[farmer_key]
                    conn = get_connection()
                    conn.execute("""INSERT INTO crops
                        (farmer_id,crop_name,field_name,area_acres,planting_date,expected_harvest,status,notes)
                        VALUES (?,?,?,?,?,?,?,?)""",
                        (fid, crop_name.strip(), field_name.strip(), area,
                         str(plant_date), str(harvest_date), status, notes.strip()))
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Crop **{crop_name}** saved!")

                    # Auto-show pest advice after saving
                    advice = get_crop_advice(crop_name)
                    st.markdown("---")
                    st.markdown(f"### 🛡️ Pest Protection Guide for **{crop_name.title()}**")
                    st.markdown(f"**Common Pests & Diseases:** {advice['common_pests']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**💊 Chemical Control:**")
                        st.info(advice['chemical'])
                        st.markdown("**🌿 Organic Control:**")
                        st.success(advice['organic'])
                    with col2:
                        st.markdown("**🌾 Cultural Practices:**")
                        st.warning(advice['cultural'])
                        st.markdown("**📅 Scouting Schedule:**")
                        st.error(advice['schedule'])
                except Exception as e:
                    st.error(f"Database error: {e}")

    with tab2:
        st.subheader("All Crop Records")
        filter_farmer = st.selectbox("Filter by Farmer", ["All"]+list(farmer_options.keys()), key="cm_filter")
        try:
            conn = get_connection()
            if filter_farmer == "All":
                rows = conn.execute("""
                    SELECT c.id,f.name,c.crop_name,c.field_name,c.area_acres,
                           c.planting_date,c.expected_harvest,c.status,c.notes
                    FROM crops c JOIN farmers f ON c.farmer_id=f.id ORDER BY c.created_at DESC""").fetchall()
            else:
                fid  = farmer_options[filter_farmer]
                rows = conn.execute("""
                    SELECT c.id,f.name,c.crop_name,c.field_name,c.area_acres,
                           c.planting_date,c.expected_harvest,c.status,c.notes
                    FROM crops c JOIN farmers f ON c.farmer_id=f.id
                    WHERE c.farmer_id=? ORDER BY c.created_at DESC""", (fid,)).fetchall()
            conn.close()
            if rows:
                df = pd.DataFrame(rows, columns=["ID","Farmer","Crop","Field","Acres","Planted","Harvest","Status","Notes"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No crop records found.")
        except Exception as e:
            st.error(f"Database error: {e}")

    with tab3:
        st.subheader("🛡️ Pest Protection Advice for Any Crop")
        st.markdown("Get complete pest management guidance for **any crop in the world**.")
        query_crop = st.text_input("Enter any crop name", placeholder="e.g. rice, mango, cotton, banana, chilli...")
        if st.button("Get Pest Advice", type="primary", key="cm_advice"):
            if query_crop.strip():
                advice = get_crop_advice(query_crop)
                st.markdown(f"### 🌱 Pest Management Guide: **{query_crop.title()}**")
                st.markdown(f"**🐛 Common Pests & Diseases:** {advice['common_pests']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**💊 Chemical Control:**")
                    st.info(advice['chemical'])
                    st.markdown("**🌿 Organic/Biological Control:**")
                    st.success(advice['organic'])
                with col2:
                    st.markdown("**🌾 Cultural Practices:**")
                    st.warning(advice['cultural'])
                    st.markdown("**📅 Scouting Schedule:**")
                    st.error(advice['schedule'])