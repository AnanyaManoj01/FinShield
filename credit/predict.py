"""Credit inference helper — imported by the Streamlit page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from functools import lru_cache

from config import CREDIT_FEATURES, CREDIT_MODEL_PATH


@lru_cache(maxsize=1)
def _load_model():
    if not CREDIT_MODEL_PATH.exists():
        raise FileNotFoundError("Credit model not found. Run: python -m credit.train")
    return joblib.load(CREDIT_MODEL_PATH)


def predict_credit(feature_values):
    """feature_values: list aligned with config.CREDIT_FEATURES.
    Returns risk band and approval probability (1 - P(high risk))."""
    model = _load_model()
    df = pd.DataFrame([feature_values], columns=CREDIT_FEATURES)
    pred = int(model.predict(df)[0])          # 1 = HIGH risk
    prob_high = float(model.predict_proba(df)[0][1])
    return {
        "risk": "HIGH" if pred == 1 else "LOW",
        "approval_probability": round((1 - prob_high) * 100, 2),
    }


def predict_credit_batch(df):
    """Score a DataFrame of many applicants. Returns a copy with result columns.
    The DataFrame must contain the columns in config.CREDIT_FEATURES."""
    model = _load_model()
    X = df[CREDIT_FEATURES]
    probs_high = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["approval_probability_%"] = ((1 - probs_high) * 100).round(2)
    out["risk"] = ["HIGH" if p >= 0.5 else "LOW" for p in probs_high]
    return out


if __name__ == "__main__":
    sample = [55000.0, 12000.0, 710, 5.0, 1, 35, 1800.0, 6.0]
    print(predict_credit(sample))
