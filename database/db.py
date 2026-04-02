import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "agri.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_columns(conn, table):
    """Return list of column names for a table (empty list if table doesn't exist)."""
    try:
        c = conn.cursor()
        c.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in c.fetchall()]
    except Exception:
        return []


def _migrate(conn):
    """
    Auto-migration: fixes ALL column name mismatches caused by old code.
    Runs every startup — safe to call repeatedly.
    """
    c = conn.cursor()

    # ── FIX 1: farmers table — land_acres → land_area ────────────────────────
    farmer_cols = _get_columns(conn, "farmers")
    if "land_acres" in farmer_cols and "land_area" not in farmer_cols:
        c.executescript("""
            ALTER TABLE farmers RENAME TO _farmers_old;
            CREATE TABLE farmers (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                phone      TEXT,
                location   TEXT,
                land_area  REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO farmers (id, name, phone, location, land_area, created_at)
                SELECT id, name, phone, location, land_acres, created_at FROM _farmers_old;
            DROP TABLE _farmers_old;
        """)
        conn.commit()

    # ── FIX 2: crops table — add missing columns / rename field_name ─────────
    crop_cols = _get_columns(conn, "crops")

    # If crops table has field_name but not crop_name → full rebuild
    if "field_name" in crop_cols and "crop_name" not in crop_cols:
        # Check which extra columns exist to preserve data
        has_status = "status" in crop_cols
        has_area   = "area"   in crop_cols

        status_col   = ", status"  if has_status else ""
        status_val   = ", status"  if has_status else ""
        area_col     = ", area"    if has_area   else ""
        area_val     = ", area"    if has_area   else ""

        c.executescript(f"""
            ALTER TABLE crops RENAME TO _crops_old;
            CREATE TABLE crops (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id        INTEGER,
                crop_name        TEXT NOT NULL,
                planting_date    TEXT,
                expected_harvest TEXT,
                area             REAL,
                status           TEXT DEFAULT 'Growing',
                notes            TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id)
            );
            INSERT INTO crops
                (id, farmer_id, crop_name, planting_date, expected_harvest{area_col}{status_col}, notes, created_at)
            SELECT
                id, farmer_id, field_name, planting_date, expected_harvest{area_val}{status_val}, notes, created_at
            FROM _crops_old;
            DROP TABLE _crops_old;
        """)
        conn.commit()

    # If status column missing from crops → add it
    crop_cols = _get_columns(conn, "crops")  # re-read after possible rebuild
    if crop_cols and "status" not in crop_cols:
        c.execute("ALTER TABLE crops ADD COLUMN status TEXT DEFAULT 'Growing'")
        conn.commit()

    # ── FIX 3: soil_records — add any missing columns ────────────────────────
    soil_cols = _get_columns(conn, "soil_records")
    if soil_cols:
        for col, col_type in [("temperature", "REAL"), ("crop_type", "TEXT")]:
            if col not in soil_cols:
                try:
                    c.execute(f"ALTER TABLE soil_records ADD COLUMN {col} {col_type}")
                    conn.commit()
                except Exception:
                    pass


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ── Farmers ───────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS farmers (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            phone      TEXT,
            location   TEXT,
            land_area  REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            status           TEXT DEFAULT 'Growing',
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
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_id        INTEGER,
            image_name       TEXT,
            disease_detected TEXT,
            confidence       REAL,
            treatment        TEXT,
            detected_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (farmer_id) REFERENCES farmers(id)
        )
    """)

    conn.commit()
    _migrate(conn)   # fix any old column-name mismatches
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
        (name.strip(), str(phone).strip(), str(location).strip(),
         float(land_area) if land_area else 0.0)
    )
    conn.commit()
    conn.close()


def update_farmer(farmer_id, name, phone, location, land_area):
    conn = get_connection()
    conn.execute(
        "UPDATE farmers SET name=?, phone=?, location=?, land_area=? WHERE id=?",
        (name, str(phone), str(location),
         float(land_area) if land_area else 0.0, farmer_id)
    )
    conn.commit()
    conn.close()


def delete_farmer(farmer_id):
    conn = get_connection()
    conn.execute("DELETE FROM farmers WHERE id=?", (farmer_id,))
    conn.commit()
    conn.close()


# ── Crop CRUD ─────────────────────────────────────────────────────────────────
def add_crop(farmer_id, crop_name, planting_date, expected_harvest, area, status, notes):
    conn = get_connection()
    conn.execute(
        """INSERT INTO crops
           (farmer_id, crop_name, planting_date, expected_harvest, area, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (farmer_id, str(crop_name).strip(),
         str(planting_date), str(expected_harvest),
         float(area) if area else 0.0,
         str(status), str(notes).strip())
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


def update_crop(crop_id, crop_name, planting_date, expected_harvest, area, status, notes):
    conn = get_connection()
    conn.execute(
        """UPDATE crops SET crop_name=?, planting_date=?, expected_harvest=?,
           area=?, status=?, notes=? WHERE id=?""",
        (crop_name, str(planting_date), str(expected_harvest),
         float(area) if area else 0.0, status, notes, crop_id)
    )
    conn.commit()
    conn.close()


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
        (farmer_id, float(ph), float(nitrogen), float(phosphorus),
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