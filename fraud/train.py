"""
Train the fraud detection model.

Improvements over the original:
  - Trains on realistic NAMED features (not feature_0..19)
  - Handles class imbalance with class_weight="balanced"
  - Reports precision / recall / F1 / ROC-AUC (not just accuracy, which is
    misleading on a 96/4 split) and the confusion matrix
  - Saves metrics to JSON so the app and README can show REAL numbers
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
    FRAUD_FEATURES, FRAUD_DATA_PATH, FRAUD_MODEL_PATH,
    FRAUD_METRICS_PATH, MODELS_DIR, DATA_DIR,
)
from fraud.generate_data import generate_fraud_data


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    if FRAUD_DATA_PATH.exists():
        df = pd.read_csv(FRAUD_DATA_PATH)
    else:
        df = generate_fraud_data()
        df.to_csv(FRAUD_DATA_PATH, index=False)

    X = df[FRAUD_FEATURES]
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",   # key fix for imbalance
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    # In fraud, catching fraud (recall) matters more than overall accuracy,
    # so we lower the decision threshold from the default 0.5. This trades a
    # few false alarms for catching substantially more fraud.
    THRESHOLD = 0.30
    pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "decision_threshold": THRESHOLD,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_fraud": round(precision_score(y_test, pred), 4),
        "recall_fraud": round(recall_score(y_test, pred), 4),
        "f1_fraud": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "n_test": int(len(y_test)),
        "fraud_rate": round(float(y.mean()), 4),
    }

    print("=" * 55)
    print(" FRAUD DETECTION MODEL")
    print("=" * 55)
    print("\nConfusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))
    print("Key metrics:", json.dumps(metrics, indent=2))

    joblib.dump(model, FRAUD_MODEL_PATH)
    with open(FRAUD_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel  -> {FRAUD_MODEL_PATH}")
    print(f"Metrics-> {FRAUD_METRICS_PATH}")


if __name__ == "__main__":
    main()
