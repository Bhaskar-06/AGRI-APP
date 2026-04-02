import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "agri.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Farmers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            location TEXT,
            land_area REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Crops table
    c.execute("""
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            crop_name TEXT NOT NULL,
            planting_date TEXT,
            expected_harvest TEXT,
            area REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    # Soil records table
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            ph REAL,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            moisture REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    # Pest/Disease detection log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS pest_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id INTEGER,
            image_name TEXT,
            disease_detected TEXT,
            confidence REAL,
            treatment TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    conn.commit()
    conn.close()


# ── Farmer Functions ──────────────────────────────────────────────────────────
def get_all_farmers():
    conn = get_connection()
    rows = conn.execute("SELECT id, name, phone, location, land_area FROM farmers").fetchall()
    conn.close()
    return rows


def add_farmer(name, phone, location, land_area):
    conn = get_connection()
    conn.execute(
        "INSERT INTO farmers (name, phone, location, land_area) VALUES (?, ?, ?, ?)",
        (name, phone, location, land_area)
    )
    conn.commit()
    conn.close()


def delete_farmer(farmer_id):
    conn = get_connection()
    conn.execute("DELETE FROM farmers WHERE id = ?", (farmer_id,))
    conn.commit()
    conn.close()


# ── Crop Functions ────────────────────────────────────────────────────────────
def add_crop(farmer_id, crop_name, planting_date, expected_harvest, area, notes):
    conn = get_connection()
    conn.execute(
        """INSERT INTO crops (farmer_id, crop_name, planting_date, expected_harvest, area, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (farmer_id, crop_name, planting_date, expected_harvest, area, notes)
    )
    conn.commit()
    conn.close()


def get_crops(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute("SELECT * FROM crops WHERE farmer_id = ?", (farmer_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM crops").fetchall()
    conn.close()
    return rows


# ── Soil Functions ────────────────────────────────────────────────────────────
def add_soil_record(farmer_id, ph, nitrogen, phosphorus, potassium, moisture):
    conn = get_connection()
    conn.execute(
        """INSERT INTO soil_records (farmer_id, ph, nitrogen, phosphorus, potassium, moisture)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (farmer_id, ph, nitrogen, phosphorus, potassium, moisture)
    )
    conn.commit()
    conn.close()


def get_soil_records(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute("SELECT * FROM soil_records WHERE farmer_id = ?", (farmer_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM soil_records").fetchall()
    conn.close()
    return rows


# ── Pest Log Functions ────────────────────────────────────────────────────────
def add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment):
    conn = get_connection()
    conn.execute(
        """INSERT INTO pest_logs (farmer_id, image_name, disease_detected, confidence, treatment)
           VALUES (?, ?, ?, ?, ?)""",
        (farmer_id, image_name, disease_detected, float(confidence), treatment)
    )
    conn.commit()
    conn.close()


def get_pest_logs(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute("SELECT * FROM pest_logs WHERE farmer_id = ?", (farmer_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM pest_logs").fetchall()
    conn.close()
    return rows