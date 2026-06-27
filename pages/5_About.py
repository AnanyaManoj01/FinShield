import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import FRAUD_METRICS_PATH, CREDIT_METRICS_PATH

st.title("ℹ About FinShield")

st.markdown("""
**FinShield** is an end-to-end payment risk assessment platform with four
modules: Fraud Detection, Credit Risk & Underwriting, Prediction History, and
an Interactive Analytics Dashboard.

**Tech stack:** Python · Scikit-learn · Streamlit · SQLite · Pandas · Plotly
**Models:** Random Forest (fraud detection and credit-risk scoring)
""")

st.subheader("Model Performance (from held-out test set)")


def show(metrics_path, title, positive_label):
    if not metrics_path.exists():
        st.info(f"{title}: run training to generate metrics.")
        return
    m = json.loads(metrics_path.read_text())
    st.markdown(f"**{title}**")
    cols = st.columns(4)
    cols[0].metric("ROC-AUC", m.get("roc_auc", "—"))
    cols[1].metric(f"Recall ({positive_label})",
                   m.get("recall_fraud") or m.get("recall_high_risk", "—"))
    cols[2].metric(f"Precision ({positive_label})",
                   m.get("precision_fraud") or m.get("precision_high_risk", "—"))
    cols[3].metric("Accuracy", m.get("accuracy", "—"))


show(FRAUD_METRICS_PATH, "Fraud Detection", "fraud")
st.divider()
show(CREDIT_METRICS_PATH, "Credit Risk", "high risk")

st.caption(
    "Note: trained on a realistic synthetic dataset with named features, "
    "class imbalance, and label noise. Drop a real CSV with matching column "
    "names into data/ to retrain on production-grade data."
)
