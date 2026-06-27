import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from credit.predict import predict_credit, predict_credit_batch, _load_model
from database.db import save_prediction
from utils_explain import top_features
from config import CREDIT_FEATURES

st.title("💰 Credit Risk & Underwriting")

tab_single, tab_batch = st.tabs(["Single applicant", "Batch (upload CSV)"])

# ----------------------------------------------------------------------
# Single applicant
# ----------------------------------------------------------------------
with tab_single:
    st.write("Enter applicant details to assess credit risk.")

    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Annual Income ($)", value=55000.0, min_value=0.0)
        loan = st.number_input("Loan Amount ($)", value=12000.0, min_value=0.0)
        credit_score = st.number_input("Credit Score (300-850)", value=710.0,
                                       min_value=300.0, max_value=850.0)
        employment = st.number_input("Employment Years", value=5.0, min_value=0.0)
    with c2:
        existing = st.number_input("Existing Loans", value=1.0, min_value=0.0)
        age = st.number_input("Age", value=35, min_value=18, max_value=100)
        expense = st.number_input("Monthly Expense ($)", value=1800.0, min_value=0.0)
        account = st.number_input("Account Age (years)", value=6.0, min_value=0.0)

    if st.button("Evaluate Applicant"):
        features = [income, loan, credit_score, employment,
                    existing, age, expense, account]
        result = predict_credit(features)

        st.divider()
        if result["risk"] == "LOW":
            st.success("✅ LOW RISK")
        else:
            st.error("⚠️ HIGH RISK")
        st.metric("Approval Probability", f"{result['approval_probability']}%")

        save_prediction(module="Credit", prediction=result["risk"],
                        probability=result["approval_probability"])

        st.subheader("Top risk drivers (model importance)")
        for name, imp in top_features(_load_model(), CREDIT_FEATURES, k=3):
            st.write(f"- **{name.replace('_', ' ').title()}** — "
                     f"{imp:.1%} of model's decision weight")
        st.caption("Prediction saved to database.")

# ----------------------------------------------------------------------
# Batch upload
# ----------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV of applicants to score them all at once.")
    st.caption("Required columns: " + ", ".join(CREDIT_FEATURES))

    uploaded = st.file_uploader("Choose a CSV file", type="csv", key="credit_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in CREDIT_FEATURES if c not in df.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                scored = predict_credit_batch(df)
                n_high = int((scored["risk"] == "HIGH").sum())
                st.success(f"Scored {len(scored)} applicants — "
                           f"{n_high} flagged as high risk.")
                st.dataframe(scored, use_container_width=True)
                st.download_button(
                    "⬇ Download scored results",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="credit_scored.csv",
                    mime="text/csv",
                )

                # Option to save every scored row to the database so they
                # appear in Analytics and Prediction History.
                if st.button("💾 Save these results to database",
                             key="save_credit_batch"):
                    for _, row in scored.iterrows():
                        save_prediction(
                            module="Credit",
                            prediction=row["risk"],
                            probability=float(row["approval_probability_%"]),
                        )
                    st.success(f"Saved {len(scored)} predictions to database.")
        except Exception as e:
            st.error(f"Could not process file: {e}")
