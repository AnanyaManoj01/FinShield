# FinShield

## Intelligent Payment Risk Assessment Platform

FinShield is an end-to-end machine learning application for payment risk assessment, integrating **Fraud Detection**, **Credit Risk Assessment**, **Prediction History**, and an **Interactive Analytics Dashboard**. It combines supervised Scikit-learn classification models with a SQLite database and a multi-page Streamlit interface, demonstrating a complete workflow from user input through model inference, database storage, and analytics visualization.

---

## Features

- Fraud detection using a Random Forest classifier with imbalance handling and threshold tuning
- Credit-risk scoring with calibrated approval probabilities
- SQLite-backed prediction history with parameterized SQL queries
- Interactive analytics dashboard (Plotly)
- Multi-page Streamlit application
- Real evaluation metrics persisted and surfaced in the UI

---

## Tech Stack

| Category          | Technologies  |
| ----------------- | ------------- |
| Programming       | Python        |
| Machine Learning  | Scikit-learn  |
| Database          | SQLite        |
| Frontend          | Streamlit     |
| Data Processing   | Pandas, NumPy |
| Visualization     | Plotly        |
| Model Persistence | Joblib        |

---

## Model Performance

Evaluated on a held-out test set. For imbalanced problems, **recall and ROC-AUC matter more than accuracy** — a model that never flags fraud can still score high accuracy while catching nothing, so those metrics are reported explicitly.

**Fraud Detection** (4% fraud rate, decision threshold tuned to 0.30 to prioritize catching fraud)

| Metric | Value |
| ------ | ----- |
| ROC-AUC | 0.83 |
| Recall (fraud) | 0.59 |
| Precision (fraud) | 0.18 |
| Accuracy | 0.88 |

**Credit Risk** (25% high-risk rate)

| Metric | Value |
| ------ | ----- |
| ROC-AUC | 0.89 |
| Recall (high risk) | 0.75 |
| Precision (high risk) | 0.64 |
| Accuracy | 0.83 |

> Trained on a realistic synthetic dataset with **named, interpretable features**, genuine class imbalance, and label noise. To retrain on production data, drop a CSV with matching column names into `data/` — no code changes needed. Suggested real datasets: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

---

## Project Structure

```text
FinShield/
├── config.py                 # single source of truth: paths, DB name, feature lists
├── dashboard.py              # landing page (loads live metrics)
├── setup.py                  # one command: generate data, train, init DB
│
├── fraud/
│   ├── generate_data.py      # realistic named features + noisy labels
│   ├── train.py              # imbalance handling, threshold tuning, metrics
│   └── predict.py            # reusable inference helper
│
├── credit/
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
│
├── database/
│   └── db.py                 # init + parameterized inserts (one DB everywhere)
│
├── pages/
│   ├── 1_Fraud_Detection.py
│   ├── 2_Credit_Risk.py
│   ├── 4_Analytics.py
│   ├── 5_About.py
│   └── 6_Prediction_History.py
│
├── models/                   # generated: .pkl models + .json metrics
├── data/                     # generated: datasets
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python setup.py            # generates data, trains both models, initializes DB
streamlit run dashboard.py
```

Or run steps individually:

```bash
python -m fraud.train
python -m credit.train
streamlit run dashboard.py
```

---

## Design Notes

- **Single source of truth for config.** Paths, the database name, and feature lists live in `config.py`, so the UI input fields always match the columns the model was trained on, and every component reads/writes the same database.
- **Honest metrics.** Training persists real precision/recall/F1/ROC-AUC to JSON; the dashboard and About page load those live rather than displaying hardcoded numbers.
- **Reusable inference.** Streamlit pages import `predict_fraud` / `predict_credit` instead of duplicating model-loading logic.
