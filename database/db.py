import streamlit as st

# ─────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except KeyError:
        st.error("❌ Missing Supabase credentials in secrets.toml")
        return None
    except ImportError:
        st.error("❌ Run: pip install supabase==2.3.0")
        return None
    except Exception as e:
        st.error(f"❌ DB Connection Error: {e}")
        return None

# ─────────────────────────────────────────
# FARMER OPERATIONS
# ─────────────────────────────────────────
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
        result = db.table("farmers").select("*")\
                   .order("created_at", desc=True).execute()
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
    except:
        return False

def db_get_farmer_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("farmers")\
                   .select("id", count="exact").execute()
        return result.count or 0
    except:
        return 0

# ─────────────────────────────────────────
# CROP OPERATIONS
# ─────────────────────────────────────────
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
        query = db.table("crop_records")\
                  .select("*, farmers(name)")\
                  .order("created_at", desc=True)
        if farmer_id:
            query = query.eq("farmer_id", farmer_id)
        result = query.execute()
        return result.data or []
    except:
        return []

def db_get_crop_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("crop_records")\
                   .select("id", count="exact").execute()
        return result.count or 0
    except:
        return 0

# ─────────────────────────────────────────
# SOIL OPERATIONS
# ─────────────────────────────────────────
def db_add_soil(farmer_id, ph, nitrogen,
                phosphorus, potassium, moisture, location):
    db = get_supabase()
    if not db:
        return False
    try:
        data = {
            "farmer_id": int(farmer_id),
            "ph_level": float(ph),
            "nitrogen": float(nitrogen),
            "phosphorus": float(phosphorus),
            "potassium": float(potassium),
            "moisture": float(moisture),
            "location": location
        }
        db.table("soil_records").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Soil add error: {e}")
        return False

def db_get_soil_records():
    db = get_supabase()
    if not db:
        return []
    try:
        result = db.table("soil_records")\
                   .select("*, farmers(name)")\
                   .order("created_at", desc=True).execute()
        return result.data or []
    except:
        return []

def db_get_soil_count():
    db = get_supabase()
    if not db:
        return 0
    try:
        result = db.table("soil_records")\
                   .select("id", count="exact").execute()
        return result.count or 0
    except:
        return 0