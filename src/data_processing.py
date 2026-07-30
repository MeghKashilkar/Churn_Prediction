"""
Data loading, cleaning, feature engineering and preprocessing pipeline
for the Telco Customer Churn dataset.

Dataset schema (classic IBM / Kaggle "Telco Customer Churn" — blastchar):
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn

This module has no dependency on the model type — it only prepares data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

ID_COL = "customerID"
TARGET_COL = "Churn"

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",  # stored as 0/1 in the raw data but is really a category
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_add_on_services",
]

ADD_ON_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find raw data at {path}. See README.md for the download "
            "instructions (Kaggle / IBM GitHub mirror)."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: fix dtypes, handle the known blank-TotalCharges rows,
    normalize the SeniorCitizen flag, and engineer a couple of features.
    """
    df = df.copy()

    # TotalCharges is read as object because ~11 rows (tenure == 0, brand new
    # customers) have a blank string instead of a number. Coerce to numeric and
    # drop those rows — they haven't been billed yet so there's no churn signal
    # to learn from them either way.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_processing] Dropped {dropped} rows with blank TotalCharges (tenure=0 new customers).")

    # SeniorCitizen ships as 0/1 int; treat it as a category like every other
    # demographic flag rather than a continuous number.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype(str)

    # Target to binary int.
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # --- Feature engineering -------------------------------------------------
    # Average spend per month of tenure. For tenure == 0 (shouldn't remain
    # after the drop above, but guard anyway) fall back to MonthlyCharges.
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Count of add-on services actively subscribed (Yes == 1). This tends to
    # be a strong, easy-to-explain churn signal: customers with more add-ons
    # are more "locked in" and churn less.
    df["num_add_on_services"] = (df[ADD_ON_SERVICE_COLS] == "Yes").sum(axis=1)

    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: one-hot encode categoricals, standardize numerics.

    Kept as its own function so training and inference always use the exact
    same transformation logic (fit once during training, saved with joblib,
    reused unchanged at inference time in the API / Streamlit app).
    """
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("num", numeric_pipeline, NUMERIC_COLS),
        ]
    )
    return preprocessor


def get_feature_columns() -> list[str]:
    return CATEGORICAL_COLS + NUMERIC_COLS


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the target so churn rate is preserved
    in both splits."""
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_and_prepare(path: str | Path):
    """Convenience wrapper: load -> clean -> split. Returns
    (X_train, X_test, y_train, y_test, full_clean_df)."""
    raw = load_raw_data(path)
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)
    return X_train, X_test, y_train, y_test, clean


def save_feature_metadata(out_path: str | Path) -> None:
    """Persist the exact column lists used at training time so the API /
    Streamlit app can validate incoming requests against the same schema."""
    metadata = {
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "raw_input_cols": [
            c for c in CATEGORICAL_COLS if c != "SeniorCitizen"
        ]
        + ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "add_on_service_cols": ADD_ON_SERVICE_COLS,
        "target_col": TARGET_COL,
    }
    Path(out_path).write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the data pipeline")
    parser.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, clean = load_and_prepare(args.data)
    print(f"Loaded {len(clean)} clean rows.")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
