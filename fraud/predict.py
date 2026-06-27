"""Fraud inference helper — imported by the Streamlit page so loading and
prediction logic lives in ONE place (the original duplicated it inline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from functools import lru_cache

from config import FRAUD_FEATURES, FRAUD_MODEL_PATH


@lru_cache(maxsize=1)
def _load_model():
    if not FRAUD_MODEL_PATH.exists():
        raise FileNotFoundError("Fraud model not found. Run: python -m fraud.train")
    return joblib.load(FRAUD_MODEL_PATH)


def predict_fraud(feature_values):
    """feature_values: list aligned with config.FRAUD_FEATURES."""
    model = _load_model()
    df = pd.DataFrame([feature_values], columns=FRAUD_FEATURES)
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "prediction": "Fraud" if pred == 1 else "Safe",
        "fraud_probability": round(prob * 100, 2),
    }


def predict_fraud_batch(df):
    """Score a DataFrame of many rows. Returns a copy with prediction columns.
    The DataFrame must contain the columns in config.FRAUD_FEATURES."""
    model = _load_model()
    X = df[FRAUD_FEATURES]
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability_%"] = (probs * 100).round(2)
    out["prediction"] = ["Fraud" if p >= 0.5 else "Safe" for p in probs]
    return out


if __name__ == "__main__":
    sample = [250.0, 75.0, 30.0, 20.0, 3, 8, 40.0, 80.0, 5.0, 1]
    print(predict_fraud(sample))
