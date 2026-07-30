"""
Train and compare several churn classifiers, pick the best one by ROC-AUC on
a held-out test set, and persist everything needed for inference:

    models/best_model.pkl        -- fitted estimator
    models/preprocessor.pkl      -- fitted ColumnTransformer
    models/feature_metadata.json -- column schema used at train time
    models/model_metadata.json   -- which model won + its metrics
    reports/model_comparison.csv -- every model's metrics, for the README/CV
    reports/figures/*.png        -- ROC curves, confusion matrix

Usage:
    python -m src.train --data data/raw/Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import build_preprocessor, load_and_prepare, save_feature_metadata

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def build_models(y_train: pd.Series) -> dict:
    """Model zoo. Every model uses class-balancing so the ~27% churn rate
    doesn't get ignored in favor of always predicting 'stays'."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate(model, X_test_t, y_test) -> dict:
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main(data_path: str, cv_folds: int = 5, test_size: float = 0.2) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading + cleaning data from {data_path} ...")
    X_train, X_test, y_train, y_test, clean_df = load_and_prepare(data_path)
    print(f"[train] Train rows: {len(X_train)} | Test rows: {len(X_test)} | Churn rate: {clean_df['Churn'].mean():.3f}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    models = build_models(y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"[train] Fitting {name} ...")
        t0 = time.time()

        cv_scores = cross_val_score(
            model, X_train_t, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_train_t, y_train)
        metrics = evaluate(model, X_test_t, y_test)
        elapsed = time.time() - t0

        row = {
            "model": name,
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            "train_seconds": round(elapsed, 2),
            **metrics,
        }
        results.append(row)
        fitted_models[name] = model
        print(
            f"[train]   {name}: test ROC-AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  CV ROC-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
        )

    comparison_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n[train] Model comparison:\n", comparison_df.to_string(index=False))

    best_row = comparison_df.iloc[0]
    best_name = best_row["model"]
    best_model = fitted_models[best_name]
    print(f"\n[train] Best model by test ROC-AUC: {best_name} ({best_row['roc_auc']:.4f})")

    # --- Persist artifacts ---------------------------------------------------
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    save_feature_metadata(MODELS_DIR / "feature_metadata.json")

    model_metadata = {
        "best_model_name": best_name,
        "metrics": {k: float(v) for k, v in best_row.items() if k != "model"},
        "all_models": comparison_df.to_dict(orient="records"),
        "trained_rows": int(len(X_train) + len(X_test)),
        "test_size": test_size,
        "cv_folds": cv_folds,
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2, default=str))

    # --- Plots -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test_t, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test_t, y_test, ax=ax, cmap="Blues", display_labels=["Stayed", "Churned"]
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\n[train] Saved model + artifacts to {MODELS_DIR}/")
    print(f"[train] Saved comparison table + figures to {REPORTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare churn models")
    parser.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    main(args.data, cv_folds=args.cv_folds, test_size=args.test_size)
