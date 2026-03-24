import sqlite3
import os

DB_PATH = "database/agriculture.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Farmer Table
    c.execute('''CREATE TABLE IF NOT EXISTS farmers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        location TEXT,
        acreage REAL,
        contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    # Crop Table
    c.execute('''CREATE TABLE IF NOT EXISTS crops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER,
        crop_type TEXT,
        field_name TEXT,
        planting_date TEXT,
        expected_harvest TEXT,
        area_acres REAL,
        notes TEXT,
        FOREIGN KEY (farmer_id) REFERENCES farmers(id)
    )''')

    # Soil Records Table
    c.execute('''CREATE TABLE IF NOT EXISTS soil_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER,
        field_name TEXT,
        nitrogen REAL,
        phosphorus REAL,
        potassium REAL,
        ph REAL,
        moisture REAL,
        temperature REAL,
        humidity REAL,
        rainfall REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (farmer_id) REFERENCES farmers(id)
    )''')

    # Pest Detection Logs
    c.execute('''CREATE TABLE IF NOT EXISTS pest_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER,
        image_name TEXT,
        detected_disease TEXT,
        confidence REAL,
        recommendation TEXT,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (farmer_id) REFERENCES farmers(id)
    )''')

    conn.commit()
    conn.close()

# --- Farmer CRUD ---
def add_farmer(name, location, acreage, contact):
    conn = get_connection()
    conn.execute("INSERT INTO farmers (name, location, acreage, contact) VALUES (?,?,?,?)",
                 (name, location, acreage, contact))
    conn.commit(); conn.close()

def get_all_farmers():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM farmers ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows

def get_farmer_by_id(fid):
    conn = get_connection()
    row = conn.execute("SELECT * FROM farmers WHERE id=?", (fid,)).fetchone()
    conn.close()
    return row

# --- Crop CRUD ---
def add_crop(farmer_id, crop_type, field_name, planting_date, expected_harvest, area_acres, notes):
    conn = get_connection()
    conn.execute("""INSERT INTO crops 
        (farmer_id, crop_type, field_name, planting_date, expected_harvest, area_acres, notes)
        VALUES (?,?,?,?,?,?,?)""",
        (farmer_id, crop_type, field_name, planting_date, expected_harvest, area_acres, notes))
    conn.commit(); conn.close()

def get_crops_by_farmer(farmer_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM crops WHERE farmer_id=?", (farmer_id,)).fetchall()
    conn.close()
    return rows

# --- Soil CRUD ---
def add_soil_record(farmer_id, field_name, N, P, K, ph, moisture, temp, humidity, rainfall):
    conn = get_connection()
    conn.execute("""INSERT INTO soil_records 
        (farmer_id, field_name, nitrogen, phosphorus, potassium, ph, moisture, temperature, humidity, rainfall)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (farmer_id, field_name, N, P, K, ph, moisture, temp, humidity, rainfall))
    conn.commit(); conn.close()

def get_soil_by_farmer(farmer_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM soil_records WHERE farmer_id=?", (farmer_id,)).fetchall()
    conn.close()
    return rows

# --- Pest Log ---
def add_pest_log(farmer_id, image_name, disease, confidence, recommendation):
    conn = get_connection()
    conn.execute("""INSERT INTO pest_logs 
        (farmer_id, image_name, detected_disease, confidence, recommendation)
        VALUES (?,?,?,?,?)""",
        (farmer_id, image_name, disease, confidence, recommendation))
    conn.commit(); conn.close()