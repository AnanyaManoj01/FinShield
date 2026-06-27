"""
Generate a realistic fraud dataset with NAMED, interpretable features.

Why this exists:
The original project trained on `feature_0..feature_19` from
make_classification, but the UI asked users to type in "Merchant Risk Score",
"IP Reputation", etc. The model never saw those concepts, so predictions were
meaningless. Here, every feature is real, on a sensible scale, and the fraud
label follows a defensible rule PLUS noise (so the problem isn't trivially
separable — exactly like real fraud data).

To use a real Kaggle dataset later: drop a CSV with the same column names
into data/fraud_data.csv and skip this generator. Nothing else changes.
"""

import numpy as np
import pandas as pd

from config import FRAUD_FEATURES, FRAUD_DATA_PATH, DATA_DIR


def generate_fraud_data(n_samples: int = 20000, fraud_rate: float = 0.04,
                        seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    transaction_amount = rng.lognormal(mean=4.0, sigma=1.1, size=n_samples)
    merchant_risk_score = rng.beta(2, 5, size=n_samples) * 100
    device_trust_score = rng.beta(5, 2, size=n_samples) * 100
    customer_account_age_days = rng.exponential(scale=400, size=n_samples)
    prev_failed_transactions = rng.poisson(lam=0.4, size=n_samples)
    transaction_velocity_24h = rng.poisson(lam=3, size=n_samples)
    ip_reputation_score = rng.beta(5, 2, size=n_samples) * 100
    geolocation_risk = rng.beta(2, 5, size=n_samples) * 100
    avg_amount = np.maximum(rng.lognormal(mean=4.0, sigma=0.6, size=n_samples), 1)
    amount_to_avg_ratio = transaction_amount / avg_amount
    is_new_device = rng.binomial(1, 0.15, size=n_samples)

    df = pd.DataFrame({
        "transaction_amount": transaction_amount,
        "merchant_risk_score": merchant_risk_score,
        "device_trust_score": device_trust_score,
        "customer_account_age_days": customer_account_age_days,
        "prev_failed_transactions": prev_failed_transactions,
        "transaction_velocity_24h": transaction_velocity_24h,
        "ip_reputation_score": ip_reputation_score,
        "geolocation_risk": geolocation_risk,
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "is_new_device": is_new_device,
    })

    # Build a fraud "risk score" from sensible relationships, then add noise.
    z = (
        0.020 * (merchant_risk_score - 30)
        + 0.020 * (geolocation_risk - 30)
        - 0.015 * (device_trust_score - 70)
        - 0.015 * (ip_reputation_score - 70)
        + 0.60 * prev_failed_transactions
        + 0.25 * (transaction_velocity_24h - 3)
        + 0.40 * np.log1p(amount_to_avg_ratio)
        + 0.80 * is_new_device
        - 0.0015 * customer_account_age_days
    )
    z = z + rng.normal(0, 1.5, size=n_samples)          # irreducible noise
    prob = 1 / (1 + np.exp(-z))

    # Calibrate threshold to hit the desired fraud rate
    threshold = np.quantile(prob, 1 - fraud_rate)
    df["fraud"] = (prob >= threshold).astype(int)

    return df[FRAUD_FEATURES + ["fraud"]]


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    data = generate_fraud_data()
    data.to_csv(FRAUD_DATA_PATH, index=False)
    print(f"Fraud dataset saved -> {FRAUD_DATA_PATH}")
    print(f"Shape: {data.shape}")
    print("Class balance:")
    print(data['fraud'].value_counts(normalize=True).round(4))
