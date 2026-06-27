import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3
import pandas as pd
import streamlit as st

from config import DB_PATH

st.title("📜 Prediction History")

conn = sqlite3.connect(DB_PATH)
try:
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY created_at DESC", conn
    )
except Exception:
    df = pd.DataFrame()
finally:
    conn.close()

if df.empty:
    st.info("No predictions yet.")
else:
    st.dataframe(df, use_container_width=True)
