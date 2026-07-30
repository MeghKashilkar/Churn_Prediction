"""
Shared inference logic used by both the FastAPI service (api/main.py) and the
Streamlit app (streamlit_app/app.py), so the two surfaces can never drift out
of sync on how a raw customer record turns into a prediction + explanation.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.data_processing import ADD_ON_SERVICE_COLS, get_feature_columns
from src.explain import build_shap_explainer, get_encoded_feature_names

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RISK_TIERS = [
    (0.7, "High"),
    (0.4, "Medium"),
    (0.0, "Low"),
]


class ModelNotTrainedError(RuntimeError):
    """Raised when models/best_model.pkl doesn't exist yet — run src/train.py first."""


@functools.lru_cache(maxsize=1)
def load_artifacts():
    """Load model + preprocessor + metadata once per process. Cached so the
    API doesn't re-read pickles on every request and Streamlit doesn't
    re-load on every rerun."""
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    metadata_path = MODELS_DIR / "model_metadata.json"

    if not (model_path.exists() and preprocessor_path.exists()):
        raise ModelNotTrainedError(
            "No trained model found in models/. Run `python -m src.train --data "
            "data/raw/Telco-Customer-Churn.csv` first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    model_name = metadata.get("best_model_name", type(model).__name__)

    feature_names = get_encoded_feature_names(preprocessor)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "model_name": model_name,
        "feature_names": feature_names,
    }


@functools.lru_cache(maxsize=1)
def get_explainer():
    """Build (and cache) the SHAP explainer. Tree models are near-instant;
    non-tree models need a small background sample which we pull from the
    training data's typical ranges via the preprocessor's fitted stats."""
    artifacts = load_artifacts()
    model, preprocessor, model_name = (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["model_name"],
    )
    # A synthetic all-zeros (post-scaling mean) background row is enough for
    # TreeExplainer (it ignores it); for linear models we fall back to a
    # handful of random points around 0 in the transformed space.
    n_features = len(artifacts["feature_names"])
    dummy_background = np.zeros((10, n_features))
    return build_shap_explainer(model, dummy_background, model_name)


def engineer_features(raw: dict[str, Any]) -> pd.DataFrame:
    """Apply the same feature engineering used at training time to a single
    raw customer record (a dict matching the API's CustomerInput schema)."""
    row = dict(raw)
    row["SeniorCitizen"] = "Yes" if str(row.get("SeniorCitizen")) in ("1", "Yes", "True", "true") else "No"

    tenure = float(row["tenure"])
    total_charges = float(row["TotalCharges"])
    monthly_charges = float(row["MonthlyCharges"])
    row["avg_monthly_spend"] = total_charges / tenure if tenure > 0 else monthly_charges
    row["num_add_on_services"] = sum(1 for c in ADD_ON_SERVICE_COLS if row.get(c) == "Yes")

    df = pd.DataFrame([row])
    return df[get_feature_columns()]


def risk_tier(probability: float) -> str:
    for threshold, label in RISK_TIERS:
        if probability >= threshold:
            return label
    return "Low"


def engineer_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of `engineer_features` for a whole DataFrame of
    already-normalized customer records (see src/schema_mapping.py)."""
    out = df.copy()
    out["SeniorCitizen"] = (
        out["SeniorCitizen"].astype(str).isin(["1", "1.0", "Yes", "True", "true"]).map({True: "Yes", False: "No"})
    )

    tenure = pd.to_numeric(out["tenure"], errors="coerce")
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["avg_monthly_spend"] = np.where(tenure > 0, total / tenure.replace(0, np.nan), monthly)
    out["num_add_on_services"] = (out[ADD_ON_SERVICE_COLS] == "Yes").sum(axis=1)

    return out[get_feature_columns()]


def predict_churn_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a whole DataFrame in a single model call.

    Returns a DataFrame with `churn_probability`, `churn_prediction` and
    `risk_tier`, indexed to match the input. SHAP is intentionally skipped
    here — per-row explanations are what made the old row-by-row loop slow,
    and they aren't useful in a bulk export anyway.
    """
    artifacts = load_artifacts()
    model, preprocessor = artifacts["model"], artifacts["preprocessor"]

    X = engineer_features_frame(df)
    X_t = preprocessor.transform(X)
    probabilities = model.predict_proba(X_t)[:, 1]

    return pd.DataFrame(
        {
            "churn_probability": probabilities.round(4),
            "churn_prediction": probabilities >= 0.5,
            "risk_tier": [risk_tier(p) for p in probabilities],
        },
        index=df.index,
    )


def predict_churn(raw: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """End-to-end: raw customer dict -> prediction + probability + top SHAP
    factors driving that specific prediction (local explainability)."""
    artifacts = load_artifacts()
    model, preprocessor, feature_names = (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["feature_names"],
    )

    X = engineer_features(raw)
    X_t = preprocessor.transform(X)

    probability = float(model.predict_proba(X_t)[0, 1])
    prediction = bool(probability >= 0.5)

    explainer = get_explainer()
    shap_out = explainer(X_t)
    values = shap_out.values
    if values.ndim == 3:
        values = values[:, :, 1]
    row_shap = values[0]

    order = np.argsort(np.abs(row_shap))[::-1][:top_k]
    top_factors = [
        {
            "feature": feature_names[i],
            "shap_value": float(row_shap[i]),
            "direction": "increases risk" if row_shap[i] > 0 else "decreases risk",
        }
        for i in order
    ]

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier(probability),
        "model_name": artifacts["model_name"],
        "top_factors": top_factors,
    }
