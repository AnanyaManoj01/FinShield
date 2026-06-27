"""
SQLite database layer.

Both initialization and inserts use DB_PATH from config — the single source
of truth. This fixes the original bug where save_prediction wrote to
FinShield.db while the analytics page read from payflow.db.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            module      TEXT    NOT NULL,
            prediction  TEXT    NOT NULL,
            probability REAL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(module: str, prediction: str, probability: float):
    """Insert one prediction. Creates the table if it doesn't exist yet."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (module, prediction, probability) VALUES (?, ?, ?)",
        (module, prediction, float(probability)),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Management helpers (view / delete / clear)
# ----------------------------------------------------------------------

def fetch_all():
    """Return all predictions as a list of dict rows, newest first."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC, id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_predictions():
    """Return total number of rows."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return n


def delete_by_id(row_id: int):
    """Delete a single prediction by its id. Returns rows deleted (0 or 1)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id = ?", (int(row_id),))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


def delete_by_module(module: str):
    """Delete all predictions for one module ('Fraud' or 'Credit')."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE module = ?", (module,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


def clear_all():
    """Delete every prediction and reset the auto-increment id counter."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    # Reset the AUTOINCREMENT counter so ids start at 1 again
    cur.execute("DELETE FROM sqlite_sequence WHERE name = 'predictions'")
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
