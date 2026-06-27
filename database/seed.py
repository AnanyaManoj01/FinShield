"""
Seed the database with sample predictions so the Analytics dashboard shows
charts immediately on first launch (instead of an empty state).

Run:  python -m database.seed
Or it's called automatically by setup.py.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.db import init_db, save_prediction


def seed(n=40, seed=42):
    random.seed(seed)
    init_db()
    for _ in range(n):
        if random.random() < 0.5:
            # Fraud module: ~15% flagged as fraud
            is_fraud = random.random() < 0.15
            save_prediction(
                module="Fraud",
                prediction="Fraud" if is_fraud else "Safe",
                probability=round(random.uniform(55, 95) if is_fraud
                                  else random.uniform(1, 30), 2),
            )
        else:
            # Credit module: ~30% high risk
            is_high = random.random() < 0.30
            save_prediction(
                module="Credit",
                prediction="HIGH" if is_high else "LOW",
                probability=round(random.uniform(20, 55) if is_high
                                  else random.uniform(70, 96), 2),
            )
    print(f"Seeded {n} sample predictions.")


if __name__ == "__main__":
    seed()
