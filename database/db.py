import requests
import streamlit as st

# =============================================================================
# SUPABASE REST API CLIENT (no SDK — avoids gotrue/httpx dependency conflicts)
# =============================================================================

def _get_config():
    try:
        url = st.secrets["supabase"]["url"].rstrip("/")
        key = st.secrets["supabase"]["key"]
        return url, key
    except Exception:
        return None, None


def is_configured():
    url, key = _get_config()
    return bool(url and key)


def _headers(prefer=None):
    _, key = _get_config()
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _rest_url(table):
    url, _ = _get_config()
    return f"{url}/rest/v1/{table}"


def _creds_ok():
    url, key = _get_config()
    if not url or not key:
        st.error("❌ Missing Supabase credentials. Add them to .streamlit/secrets.toml "
                  "(locally) or to your Streamlit Cloud app's Secrets settings.")
        return False
    return True


def _get(table, params=None):
    if not _creds_ok():
        return []
    try:
        r = requests.get(_rest_url(table), headers=_headers(), params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"DB fetch error: {e}")
        return []


def _post(table, data):
    if not _creds_ok():
        return False, "No DB connection"
    try:
        r = requests.post(_rest_url(table), headers=_headers("return=minimal"), json=data, timeout=15)
        r.raise_for_status()
        return True, "Success"
    except Exception as e:
        return False, str(e)


def _patch(table, row_id, data):
    if not _creds_ok():
        return False
    try:
        r = requests.patch(
            _rest_url(table), headers=_headers("return=minimal"),
            params={"id": f"eq.{row_id}"}, json=data, timeout=15
        )
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Update error: {e}")
        return False


def _delete(table, row_id):
    if not _creds_ok():
        return False
    try:
        r = requests.delete(_rest_url(table), headers=_headers(), params={"id": f"eq.{row_id}"}, timeout=15)
        r.raise_for_status()
        return True
    except Exception:
        return False


def _count(table):
    if not _creds_ok():
        return 0
    try:
        r = requests.get(
            _rest_url(table),
            headers=_headers("count=exact"),
            params={"select": "id", "limit": 1},
            timeout=15
        )
        r.raise_for_status()
        content_range = r.headers.get("Content-Range", "0/0")
        return int(content_range.split("/")[-1])
    except Exception:
        return 0


# =============================================================================
# FARMER FUNCTIONS
# =============================================================================

def db_add_farmer(name, location, land_acres, contact):
    data = {"name": name, "location": location, "land_acres": float(land_acres), "contact": contact}
    return _post("farmers", data)


def db_get_farmers():
    return _get("farmers", {"select": "*", "order": "created_at.desc"})


def db_delete_farmer(farmer_id):
    return _delete("farmers", farmer_id)


def db_get_farmer_count():
    return _count("farmers")


def get_all_farmers():
    return db_get_farmers()


# =============================================================================
# CROP FUNCTIONS
# =============================================================================

def db_add_crop(farmer_id, crop_name, planting_date, harvest_date, area_acres, status, notes=""):
    data = {
        "farmer_id": int(farmer_id), "crop_name": crop_name,
        "planting_date": str(planting_date), "harvest_date": str(harvest_date),
        "area_acres": float(area_acres), "status": status, "notes": notes
    }
    success, _ = _post("crop_records", data)
    return success


def db_get_crops(farmer_id=None):
    params = {"select": "*,farmers(name)", "order": "created_at.desc"}
    if farmer_id:
        params["farmer_id"] = f"eq.{farmer_id}"
    return _get("crop_records", params)


def db_update_crop(crop_id, crop_name, status, notes=""):
    return _patch("crop_records", crop_id, {"crop_name": crop_name, "status": status, "notes": notes})


def db_delete_crop(crop_id):
    return _delete("crop_records", crop_id)


def db_get_crop_count():
    return _count("crop_records")


def add_crop(farmer_id, crop_name, planting_date, harvest_date, area_acres, status, notes=""):
    return db_add_crop(farmer_id, crop_name, planting_date, harvest_date, area_acres, status, notes)

def get_crops(farmer_id=None):
    return db_get_crops(farmer_id)

def update_crop(crop_id, crop_name, status, notes=""):
    return db_update_crop(crop_id, crop_name, status, notes)

def delete_crop(crop_id):
    return db_delete_crop(crop_id)


# =============================================================================
# SOIL FUNCTIONS
# =============================================================================

def db_add_soil(farmer_id, field_name, nitrogen, phosphorus, potassium,
                 ph, temperature, humidity, moisture, rainfall, recommended_crop=None):
    data = {
        "farmer_id": int(farmer_id) if farmer_id else None,
        "field_name": field_name,
        "nitrogen": float(nitrogen), "phosphorus": float(phosphorus), "potassium": float(potassium),
        "ph_level": float(ph), "temperature": float(temperature), "humidity": float(humidity),
        "moisture": float(moisture), "rainfall": float(rainfall), "recommended_crop": recommended_crop,
    }
    success, _ = _post("soil_records", data)
    return success


def db_get_soil_records(farmer_id=None):
    params = {"select": "*,farmers(name)", "order": "created_at.desc"}
    if farmer_id:
        params["farmer_id"] = f"eq.{farmer_id}"
    return _get("soil_records", params)


def db_get_soil_count():
    return _count("soil_records")


# =============================================================================
# PEST LOG FUNCTIONS
# =============================================================================

def db_add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied=""):
    data = {
        "farmer_id": int(farmer_id) if farmer_id else None,
        "disease_detected": disease_detected, "confidence": float(confidence),
        "image_name": image_name, "treatment_applied": treatment_applied,
    }
    success, _ = _post("pest_detection_logs", data)
    return success


def db_get_pest_logs(farmer_id=None):
    params = {"select": "*,farmers(name)", "order": "created_at.desc"}
    if farmer_id:
        params["farmer_id"] = f"eq.{farmer_id}"
    return _get("pest_detection_logs", params)


def add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied=""):
    return db_add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment_applied)


# =============================================================================
# SOIL IMAGE LOG FUNCTIONS
# =============================================================================

def db_add_soil_image_log(farmer_id, image_name, soil_type, confidence):
    data = {
        "farmer_id": int(farmer_id) if farmer_id else None,
        "image_name": image_name, "soil_type": soil_type, "confidence": float(confidence),
    }
    success, _ = _post("soil_image_logs", data)
    return success


def db_get_soil_image_logs(farmer_id=None):
    params = {"select": "*,farmers(name)", "order": "created_at.desc"}
    if farmer_id:
        params["farmer_id"] = f"eq.{farmer_id}"
    return _get("soil_image_logs", params)