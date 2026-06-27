"""
Central configuration for FinShield.

Defining paths and the database name in ONE place fixes the bug where
different files pointed at different .db files (FinShield.db vs payflow.db),
which caused predictions to be saved to one database while analytics read
from another (empty) one.
"""

from pathlib import Path

# Project root = folder containing this file
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Single source of truth for the database location
DB_PATH = ROOT / "finshield.db"

# Model artifacts
FRAUD_MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
CREDIT_MODEL_PATH = MODELS_DIR / "credit_model.pkl"

# Datasets
FRAUD_DATA_PATH = DATA_DIR / "fraud_data.csv"
CREDIT_DATA_PATH = DATA_DIR / "credit_data.csv"

# Saved evaluation metrics (so the README / About page can show REAL numbers)
FRAUD_METRICS_PATH = MODELS_DIR / "fraud_metrics.json"
CREDIT_METRICS_PATH = MODELS_DIR / "credit_metrics.json"

# Feature definitions — used by both training and the UI so the input
# fields always match what the model was actually trained on.
FRAUD_FEATURES = [
    "transaction_amount",
    "merchant_risk_score",
    "device_trust_score",
    "customer_account_age_days",
    "prev_failed_transactions",
    "transaction_velocity_24h",
    "ip_reputation_score",
    "geolocation_risk",
    "amount_to_avg_ratio",
    "is_new_device",
]

CREDIT_FEATURES = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_loans",
    "age",
    "monthly_expense",
    "account_age_years",
]
