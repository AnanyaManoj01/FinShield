"""
Train the credit-risk model.

Label convention is now EXPLICIT and enforced in data generation:
    risk = 1  ->  HIGH risk
    risk = 0  ->  LOW risk
This removes the original silent assumption that "pred == 1 means high risk".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import (
    CREDIT_FEATURES, CREDIT_DATA_PATH, CREDIT_MODEL_PATH,
    CREDIT_METRICS_PATH, MODELS_DIR, DATA_DIR,
)
from credit.generate_data import generate_credit_data


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    if CREDIT_DATA_PATH.exists():
        df = pd.read_csv(CREDIT_DATA_PATH)
    else:
        df = generate_credit_data()
        df.to_csv(CREDIT_DATA_PATH, index=False)

    X = df[CREDIT_FEATURES]
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_high_risk": round(precision_score(y_test, pred), 4),
        "recall_high_risk": round(recall_score(y_test, pred), 4),
        "f1_high_risk": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "n_test": int(len(y_test)),
        "high_risk_rate": round(float(y.mean()), 4),
    }

    print("=" * 55)
    print(" CREDIT RISK MODEL")
    print("=" * 55)
    print("\nConfusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))
    print("Key metrics:", json.dumps(metrics, indent=2))

    joblib.dump(model, CREDIT_MODEL_PATH)
    with open(CREDIT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel  -> {CREDIT_MODEL_PATH}")
    print(f"Metrics-> {CREDIT_METRICS_PATH}")


if __name__ == "__main__":
    main()
