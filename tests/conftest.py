"""
Shared pytest fixtures.

The API/predict tests need a trained model on disk (models/best_model.pkl +
preprocessor.pkl). If you've already run `python -m src.train`, these fixtures
are a no-op and your real model gets exercised. If not (e.g. a fresh CI
checkout), this bootstraps a tiny model from synthetic data so the test suite
is self-contained and doesn't require the full dataset or a long training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"


def _make_synthetic_raw_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.6, 0.25, 0.15])
    tenure = rng.randint(0, 72, size=n)
    monthly = np.round(rng.uniform(18, 120, size=n), 2)
    total = np.round(monthly * np.maximum(tenure, 1) + rng.normal(0, 20, size=n), 2)

    # Bias churn towards month-to-month + low tenure, like the real dataset.
    churn_prob = 0.05 + 0.5 * (contracts == "Month-to-month") + 0.3 * (tenure < 6)
    churn = rng.binomial(1, np.clip(churn_prob, 0, 0.95))

    df = pd.DataFrame(
        {
            "customerID": [f"synthetic-{i}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], size=n),
            "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], size=n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": contracts,
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return df


@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained(tmp_path_factory):
    if (MODELS_DIR / "best_model.pkl").exists() and (MODELS_DIR / "preprocessor.pkl").exists():
        yield
        return

    import joblib
    from sklearn.linear_model import LogisticRegression

    from src.data_processing import build_preprocessor, clean_data, save_feature_metadata, split_data

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = _make_synthetic_raw_df()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_t, y_train)

    joblib.dump(model, MODELS_DIR / "best_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    save_feature_metadata(MODELS_DIR / "feature_metadata.json")
    (MODELS_DIR / "model_metadata.json").write_text(
        '{"best_model_name": "logistic_regression (synthetic test bootstrap)"}'
    )

    yield
