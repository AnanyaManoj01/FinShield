import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from database.db import (
    fetch_all, count_predictions, delete_by_id,
    delete_by_module, clear_all, save_prediction,
)

st.title("🗄️ Database Manager")
st.write("View, add, and delete prediction records stored in SQLite.")

total = count_predictions()
st.metric("Total records in database", total)

st.divider()

# ----------------------------------------------------------------------
# View all records
# ----------------------------------------------------------------------
st.subheader("📋 All records")
rows = fetch_all()
if not rows:
    st.info("Database is empty.")
else:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇ Export all records (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="finshield_predictions.csv",
        mime="text/csv",
    )

st.divider()

# ----------------------------------------------------------------------
# Add a manual record
# ----------------------------------------------------------------------
st.subheader("➕ Add a record manually")
c1, c2, c3 = st.columns(3)
with c1:
    add_module = st.selectbox("Module", ["Fraud", "Credit"], key="add_mod")
with c2:
    if add_module == "Fraud":
        add_pred = st.selectbox("Prediction", ["Fraud", "Safe"], key="add_pred")
    else:
        add_pred = st.selectbox("Prediction", ["HIGH", "LOW"], key="add_pred")
with c3:
    add_prob = st.number_input("Probability (%)", value=50.0,
                               min_value=0.0, max_value=100.0, key="add_prob")

if st.button("Add record"):
    save_prediction(module=add_module, prediction=add_pred, probability=add_prob)
    st.success("Record added.")
    st.rerun()

st.divider()

# ----------------------------------------------------------------------
# Delete a single record by id
# ----------------------------------------------------------------------
st.subheader("🗑️ Delete a single record")
if rows:
    ids = [r["id"] for r in rows]
    del_id = st.selectbox("Select record id to delete", ids, key="del_id")
    if st.button("Delete this record"):
        n = delete_by_id(del_id)
        if n:
            st.success(f"Deleted record #{del_id}.")
        else:
            st.warning("No record found with that id.")
        st.rerun()
else:
    st.caption("Nothing to delete.")

st.divider()

# ----------------------------------------------------------------------
# Delete by module
# ----------------------------------------------------------------------
st.subheader("🧹 Delete all records of one type")
c1, c2 = st.columns(2)
with c1:
    if st.button("Delete all Fraud records"):
        n = delete_by_module("Fraud")
        st.success(f"Deleted {n} fraud records.")
        st.rerun()
with c2:
    if st.button("Delete all Credit records"):
        n = delete_by_module("Credit")
        st.success(f"Deleted {n} credit records.")
        st.rerun()

st.divider()

# ----------------------------------------------------------------------
# Clear everything (with confirmation)
# ----------------------------------------------------------------------
st.subheader("⚠️ Clear the entire database")
st.caption("This deletes every record and resets ids to start at 1. Cannot be undone.")
confirm = st.checkbox("Yes, I understand this wipes all records.")
if st.button("Clear ALL records", disabled=not confirm):
    n = clear_all()
    st.success(f"Database cleared. Removed {n} records.")
    st.rerun()
