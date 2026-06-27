"""
Generate a realistic credit-risk dataset with NAMED features.

The label (high vs low risk) follows a defensible relationship to income,
credit score, debt load, etc., plus noise — so it isn't trivially separable.
Swap in a real dataset (German Credit, Home Credit) with the same column
names later and nothing downstream changes.
"""

import numpy as np
import pandas as pd

from config import CREDIT_FEATURES, CREDIT_DATA_PATH, DATA_DIR


def generate_credit_data(n_samples: int = 12000, high_risk_rate: float = 0.25,
                         seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_income = rng.lognormal(mean=10.9, sigma=0.5, size=n_samples)   # ~$ tens of thousands
    loan_amount = rng.lognormal(mean=9.5, sigma=0.7, size=n_samples)
    credit_score = np.clip(rng.normal(680, 70, size=n_samples), 300, 850)
    employment_years = np.clip(rng.exponential(scale=6, size=n_samples), 0, 40)
    existing_loans = rng.poisson(lam=1.2, size=n_samples)
    age = np.clip(rng.normal(38, 11, size=n_samples), 18, 75).astype(int)
    monthly_expense = annual_income / 12 * rng.uniform(0.2, 0.7, size=n_samples)
    account_age_years = np.clip(rng.exponential(scale=5, size=n_samples), 0, 30)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "age": age,
        "monthly_expense": monthly_expense,
        "account_age_years": account_age_years,
    })

    debt_to_income = loan_amount / np.maximum(annual_income, 1)
    z = (
        2.5 * debt_to_income
        - 0.012 * (credit_score - 680)
        + 0.40 * existing_loans
        - 0.05 * employment_years
        - 0.000004 * (annual_income - 50000)
        - 0.04 * account_age_years
    )
    z = z + rng.normal(0, 1.0, size=n_samples)
    prob = 1 / (1 + np.exp(-z))

    threshold = np.quantile(prob, 1 - high_risk_rate)
    df["risk"] = (prob >= threshold).astype(int)   # 1 = HIGH risk, 0 = LOW risk

    return df[CREDIT_FEATURES + ["risk"]]


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    data = generate_credit_data()
    data.to_csv(CREDIT_DATA_PATH, index=False)
    print(f"Credit dataset saved -> {CREDIT_DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(data['risk'].value_counts(normalize=True).round(4))
