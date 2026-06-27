import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

from config import DB_PATH

st.title("📊 Analytics Dashboard")

conn = sqlite3.connect(DB_PATH)
try:
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
except Exception:
    df = pd.DataFrame()
finally:
    conn.close()

if df.empty:
    st.warning("No predictions yet. Run a fraud or credit prediction first.")
    st.stop()

fraud_detected = len(df[(df["module"] == "Fraud") & (df["prediction"] == "Fraud")])
safe_txn = len(df[(df["module"] == "Fraud") & (df["prediction"] == "Safe")])
low_risk = len(df[(df["module"] == "Credit") & (df["prediction"] == "LOW")])
high_risk = len(df[(df["module"] == "Credit") & (df["prediction"] == "HIGH")])

st.subheader("Platform Overview")
k1, k2, k3 = st.columns(3)
k1.metric("Total Predictions", len(df))
k2.metric("Fraud Cases", fraud_detected)
k3.metric("Credit Decisions", low_risk + high_risk)

st.divider()

if fraud_detected + safe_txn > 0:
    fraud_df = pd.DataFrame({"Status": ["Fraud", "Safe"],
                             "Count": [fraud_detected, safe_txn]})
    st.plotly_chart(px.pie(fraud_df, values="Count", names="Status",
                           title="Fraud Prediction Distribution"),
                    use_container_width=True)

if low_risk + high_risk > 0:
    credit_df = pd.DataFrame({"Decision": ["Low Risk", "High Risk"],
                              "Count": [low_risk, high_risk]})
    st.plotly_chart(px.bar(credit_df, x="Decision", y="Count",
                           title="Credit Risk Predictions"),
                    use_container_width=True)

st.subheader("Recent Predictions")
st.dataframe(df.sort_values("created_at", ascending=False),
             use_container_width=True)
