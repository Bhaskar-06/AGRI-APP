import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "agri.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn):
    """
    Auto-migration: fixes column name mismatches in already-deployed databases.
    SQLite does not support ALTER COLUMN or RENAME COLUMN in older versions,
    so we recreate the table with correct columns when needed.
    """
    c = conn.cursor()

    # Check existing columns in farmers table
    c.execute("PRAGMA table_info(farmers)")
    cols = [row[1] for row in c.fetchall()]

    # If 'land_acres' exists but 'land_area' does not → rename via table rebuild
    if "land_acres" in cols and "land_area" not in cols:
        c.executescript("""
            ALTER TABLE farmers RENAME TO farmers_old;

            CREATE TABLE farmers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                phone       TEXT,
                location    TEXT,
                land_area   REAL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO farmers (id, name, phone, location, land_area, created_at)
            SELECT id, name, phone, location, land_acres, created_at FROM farmers_old;

            DROP TABLE farmers_old;
        """)
        conn.commit()

    # If neither column exists yet (fresh or broken) — create fresh
    elif "land_area" not in cols and "land_acres" not in cols:
        c.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                phone       TEXT,
                location    TEXT,
                land_area   REAL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ── Farmers ───────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            phone       TEXT,
            location    TEXT,
            land_area   REAL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Crops ─────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS crops (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id        INTEGER,
            crop_name        TEXT NOT NULL,
            planting_date    TEXT,
            expected_harvest TEXT,
            area             REAL,
            notes            TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    # ── Soil Records ──────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS soil_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id   INTEGER,
            ph          REAL,
            nitrogen    REAL,
            phosphorus  REAL,
            potassium   REAL,
            moisture    REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    # ── Pest / Disease Logs ───────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS pest_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id         INTEGER,
            image_name        TEXT,
            disease_detected  TEXT,
            confidence        REAL,
            treatment         TEXT,
            detected_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    conn.commit()

    # Run migration AFTER tables exist
    _migrate(conn)

    conn.close()


# ── Farmer CRUD ───────────────────────────────────────────────────────────────
def get_all_farmers():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, phone, location, land_area FROM farmers ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def add_farmer(name, phone, location, land_area):
    conn = get_connection()
    conn.execute(
        "INSERT INTO farmers (name, phone, location, land_area) VALUES (?, ?, ?, ?)",
        (name, str(phone), str(location), float(land_area) if land_area else 0.0)
    )
    conn.commit()
    conn.close()


def update_farmer(farmer_id, name, phone, location, land_area):
    conn = get_connection()
    conn.execute(
        "UPDATE farmers SET name=?, phone=?, location=?, land_area=? WHERE id=?",
        (name, str(phone), str(location), float(land_area) if land_area else 0.0, farmer_id)
    )
    conn.commit()
    conn.close()


def delete_farmer(farmer_id):
    conn = get_connection()
    conn.execute("DELETE FROM farmers WHERE id=?", (farmer_id,))
    conn.commit()
    conn.close()


# ── Crop CRUD ─────────────────────────────────────────────────────────────────
def add_crop(farmer_id, crop_name, planting_date, expected_harvest, area, notes):
    conn = get_connection()
    conn.execute(
        """INSERT INTO crops
           (farmer_id, crop_name, planting_date, expected_harvest, area, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (farmer_id, crop_name, str(planting_date), str(expected_harvest),
         float(area) if area else 0.0, str(notes))
    )
    conn.commit()
    conn.close()


def get_crops(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute(
            "SELECT * FROM crops WHERE farmer_id=? ORDER BY id DESC", (farmer_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM crops ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def delete_crop(crop_id):
    conn = get_connection()
    conn.execute("DELETE FROM crops WHERE id=?", (crop_id,))
    conn.commit()
    conn.close()


# ── Soil CRUD ─────────────────────────────────────────────────────────────────
def add_soil_record(farmer_id, ph, nitrogen, phosphorus, potassium, moisture):
    conn = get_connection()
    conn.execute(
        """INSERT INTO soil_records
           (farmer_id, ph, nitrogen, phosphorus, potassium, moisture)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (farmer_id,
         float(ph), float(nitrogen), float(phosphorus),
         float(potassium), float(moisture))
    )
    conn.commit()
    conn.close()


def get_soil_records(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute(
            "SELECT * FROM soil_records WHERE farmer_id=? ORDER BY id DESC", (farmer_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM soil_records ORDER BY id DESC").fetchall()
    conn.close()
    return rows


# ── Pest Log CRUD ─────────────────────────────────────────────────────────────
def add_pest_log(farmer_id, image_name, disease_detected, confidence, treatment):
    conn = get_connection()
    conn.execute(
        """INSERT INTO pest_logs
           (farmer_id, image_name, disease_detected, confidence, treatment)
           VALUES (?, ?, ?, ?, ?)""",
        (farmer_id, str(image_name), str(disease_detected),
         float(confidence), str(treatment))
    )
    conn.commit()
    conn.close()


def get_pest_logs(farmer_id=None):
    conn = get_connection()
    if farmer_id:
        rows = conn.execute(
            "SELECT * FROM pest_logs WHERE farmer_id=? ORDER BY id DESC", (farmer_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM pest_logs ORDER BY id DESC").fetchall()
    conn.close()
    return rows