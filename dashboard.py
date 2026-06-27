import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from config import FRAUD_METRICS_PATH, CREDIT_METRICS_PATH

st.set_page_config(page_title="FinShield", page_icon="💳", layout="wide")

st.title("💳 FinShield Platform")
st.markdown("""
### Intelligent Payment Risk Assessment Platform

An end-to-end ML platform for payment risk: fraud detection, credit-risk
underwriting, prediction history, and analytics.

**Select a module from the left sidebar** to begin.
""")

st.divider()


def metric_value(path, *keys, default="—"):
    if not path.exists():
        return default
    m = json.loads(path.read_text())
    for k in keys:
        if k in m:
            return m[k]
    return default


c1, c2, c3 = st.columns(3)
c1.metric("Fraud Model ROC-AUC", metric_value(FRAUD_METRICS_PATH, "roc_auc"))
c2.metric("Credit Model ROC-AUC", metric_value(CREDIT_METRICS_PATH, "roc_auc"))
c3.metric("Platform Status", "Active ✅")

st.caption("Metrics load live from the latest trained models. "
           "Run `python -m fraud.train` and `python -m credit.train` to refresh.")
