import streamlit as st

# =============================================================================
# SUPABASE CONNECTION
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_supabase():
    """Create (and cache) a single Supabase client for the whole app."""
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        st.error("❌ Missing Supabase credentials. Add them to .streamlit/secrets.toml "
                  "(locally) or to your Streamlit Cloud app's Secrets settings.")
        return None
    except ImportError:
        st.error("❌ Run: pip install supabase==2.3.0")
        return None
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return None


# =============================================================================
# FARMER FUNCTIONS
# =============================================================================

def db_add_farmer(name, location, land_acres, contact):
    db = get_supabase()
    if not db:
        return False, "No DB connection"
    try:
        data = {
            "name": name,
            "location": location,
            "land_acres": float(land_acres),
            "contact": contact
        }
        db.table("farmers").insert(data).execute()
        return True, "Success"
    except Exception as e:
        return False, str(e)


def db_get_farmers():
    db = get_supabase()
    if not db:
        return []
    try:
        result = db.table("farmers").select("*").order("created_at", desc=True).execute()
        return result.data or []
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return []


def db_delete_farmer(farmer_id):
    db = get_supabase()
    if not db:
        return False
    try:
        db.table("farmers").delete().eq("id", farmer_id).execute()
        return True
    except Exception:
        return False


def db_get_farmer_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("farmers").select("id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0


def get_all_farmers():
    """Alias used by plant.py for the farmer dropdown."""
    return db_get_farmers()


# =============================================================================
# CROP FUNCTIONS
# =============================================================================

def db_add_crop(farmer_id, crop_name, planting_date,
                 harvest_date, area_acres, status, notes=""):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {
            "farmer_id": int(farmer_id),
            "crop_name": crop_name,
            "planting_date": str(planting_date),
            "harvest_date": str(harvest_date),
            "area_acres": float(area_acres),
            "status": status,
            "notes": notes
        }
        db.table("crop_records").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Crop add error: {e}")
        return False


def db_get_crops(farmer_id=None):
    db = get_supabase()
    if not db:
        return []
    try:
        query = db.table("crop_records").select("*, farmers(name)").order("created_at", desc=True)
        if farmer_id:
            query = query.eq("farmer_id", farmer_id)
        result = query.execute()
        return result.data or []
    except Exception:
        return []


def db_update_crop(crop_id, crop_name, status, notes=""):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {"crop_name": crop_name, "status": status, "notes": notes}
        db.table("crop_records").update(data).eq("id", crop_id).execute()
        return True
    except Exception as e:
        st.error(f"Update error: {e}")
        return False


def db_delete_crop(crop_id):
    db = get_supabase()
    if not db:
        return False
    try:
        db.table("crop_records").delete().eq("id", crop_id).execute()
        return True
    except Exception:
        return False


def db_get_crop_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("crop_records").select("id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0


def add_crop(farmer_id, crop_name, planting_date, harvest_date, area_acres, status, notes=""):
    return db_add_crop(farmer_id, crop_name, planting_date, harvest_date, area_acres, status, notes)

def get_crops(farmer_id=None):
    return db_get_crops(farmer_id)

def update_crop(crop_id, crop_name, status, notes=""):
    return db_update_crop(crop_id, crop_name, status, notes)

def delete_crop(crop_id):
    return db_delete_crop(crop_id)


# =============================================================================
# SOIL FUNCTIONS  (extended: field_name, temperature, humidity, rainfall, recommended_crop)
# =============================================================================

def db_add_soil(farmer_id, field_name, nitrogen, phosphorus, potassium,
                 ph, temperature, humidity, moisture, rainfall, recommended_crop=None):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {
            "farmer_id": int(farmer_id) if farmer_id else None,
            "field_name": field_name,
            "nitrogen": float(nitrogen),
            "phosphorus": float(phosphorus),
            "potassium": float(potassium),
            "ph_level": float(ph),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "moisture": float(moisture),
            "rainfall": float(rainfall),
            "recommended_crop": recommended_crop,
        }
        db.table("soil_records").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Soil add error: {e}")
        return False


def db_get_soil_records(farmer_id=None):
    db = get_supabase()
    if not db:
        return []
    try:
        query = db.table("soil_records").select("*, farmers(name)").order("created_at", desc=True)
        if farmer_id:
            query = query.eq("farmer_id", farmer_id)
        result = query.execute()
        return result.data or []
    except Exception:
        return []


def db_get_soil_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("soil_records").select("id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0


# =============================================================================
# PEST / DISEASE DETECTION LOG FUNCTIONS
# =============================================================================

def db_add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied=""):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {
            "farmer_id": int(farmer_id) if farmer_id else None,
            "disease_detected": disease_detected,
            "confidence": float(confidence),
            "image_name": image_name,
            "treatment_applied": treatment_applied
        }
        db.table("pest_detection_logs").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Pest log error: {e}")
        return False


def db_get_pest_logs(farmer_id=None):
    db = get_supabase()
    if not db:
        return []
    try:
        query = db.table("pest_detection_logs").select("*, farmers(name)").order("created_at", desc=True)
        if farmer_id:
            query = query.eq("farmer_id", farmer_id)
        result = query.execute()
        return result.data or []
    except Exception:
        return []


def add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied=""):
    return db_add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied)


# =============================================================================
# SOIL IMAGE CLASSIFICATION LOG FUNCTIONS
# =============================================================================

def db_add_soil_image_log(farmer_id, image_name, soil_type, confidence):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {
            "farmer_id": int(farmer_id) if farmer_id else None,
            "image_name": image_name,
            "soil_type": soil_type,
            "confidence": float(confidence),
        }
        db.table("soil_image_logs").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Soil image log error: {e}")
        return False


def db_get_soil_image_logs(farmer_id=None):
    db = get_supabase()
    if not db:
        return []
    try:
        query = db.table("soil_image_logs").select("*, farmers(name)").order("created_at", desc=True)
        if farmer_id:
            query = query.eq("farmer_id", farmer_id)
        result = query.execute()
        return result.data or []
    except Exception:
        return []