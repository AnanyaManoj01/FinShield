import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from fraud.predict import predict_fraud, predict_fraud_batch, _load_model
from database.db import save_prediction
from utils_explain import top_features
from config import FRAUD_FEATURES

st.title("🛡 Fraud Detection Engine")

tab_single, tab_batch = st.tabs(["Single transaction", "Batch (upload CSV)"])

# ----------------------------------------------------------------------
# Single transaction
# ----------------------------------------------------------------------
with tab_single:
    st.write("Enter transaction details to assess fraud risk.")

    # Label -> (default, min, max). Order matches config.FRAUD_FEATURES.
    fields = [
        ("Transaction Amount ($)",        250.0,  0.0,   100000.0),
        ("Merchant Risk Score (0-100)",   30.0,   0.0,   100.0),
        ("Device Trust Score (0-100)",    70.0,   0.0,   100.0),
        ("Customer Account Age (days)",   400.0,  0.0,   5000.0),
        ("Previous Failed Transactions",  0.0,    0.0,   50.0),
        ("Transaction Velocity (24h)",    3.0,    0.0,   100.0),
        ("IP Reputation Score (0-100)",   70.0,   0.0,   100.0),
        ("Geolocation Risk (0-100)",      30.0,   0.0,   100.0),
        ("Amount-to-Average Ratio",       1.0,    0.0,   100.0),
        ("Is New Device (1=yes, 0=no)",   0.0,    0.0,   1.0),
    ]

    values = []
    c1, c2 = st.columns(2)
    for i, (label, default, lo, hi) in enumerate(fields):
        col = c1 if i % 2 == 0 else c2
        with col:
            values.append(st.number_input(label, value=default, min_value=lo,
                                          max_value=hi, key=f"f{i}"))

    if st.button("🔍 Predict Fraud"):
        result = predict_fraud(values)

        st.divider()
        if result["prediction"] == "Fraud":
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Safe Transaction")
        st.metric("Fraud Probability", f"{result['fraud_probability']:.2f}%")

        save_prediction(module="Fraud", prediction=result["prediction"],
                        probability=result["fraud_probability"])

        st.subheader("Top risk drivers (model importance)")
        for name, imp in top_features(_load_model(), FRAUD_FEATURES, k=3):
            st.write(f"- **{name.replace('_', ' ').title()}** — "
                     f"{imp:.1%} of model's decision weight")
        st.caption("Prediction saved to database.")

# ----------------------------------------------------------------------
# Batch upload
# ----------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV of transactions to score them all at once.")
    st.caption("Required columns: " + ", ".join(FRAUD_FEATURES))

    uploaded = st.file_uploader("Choose a CSV file", type="csv", key="fraud_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in FRAUD_FEATURES if c not in df.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                scored = predict_fraud_batch(df)
                n_fraud = int((scored["prediction"] == "Fraud").sum())
                st.success(f"Scored {len(scored)} transactions — "
                           f"{n_fraud} flagged as fraud.")
                st.dataframe(scored, use_container_width=True)
                st.download_button(
                    "⬇ Download scored results",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_scored.csv",
                    mime="text/csv",
                )

                # Option to save every scored row to the database so they
                # appear in Analytics and Prediction History.
                if st.button("💾 Save these results to database",
                             key="save_fraud_batch"):
                    for _, row in scored.iterrows():
                        save_prediction(
                            module="Fraud",
                            prediction=row["prediction"],
                            probability=float(row["fraud_probability_%"]),
                        )
                    st.success(f"Saved {len(scored)} predictions to database.")
        except Exception as e:
            st.error(f"Could not process file: {e}")
