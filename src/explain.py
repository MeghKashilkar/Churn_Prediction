"""
Global SHAP explainability report for the trained model.

Run after src/train.py has produced models/best_model.pkl + preprocessor.pkl.
Produces:
    reports/figures/shap_summary_bar.png     -- mean |SHAP| per feature
    reports/figures/shap_summary_beeswarm.png -- distribution of impact per feature
    reports/shap_top_features.json           -- ranked feature importances

Usage:
    python -m src.explain --data data/raw/Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.data_processing import build_preprocessor, load_and_prepare

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TREE_MODELS = ("random_forest", "gradient_boosting", "xgboost")


def get_encoded_feature_names(preprocessor) -> list[str]:
    """Recover human-readable feature names after ColumnTransformer's
    one-hot + scaling transforms, so SHAP plots show 'Contract_Two year'
    instead of 'x14'."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = preprocessor.transformers_[0][2]
    num_cols = preprocessor.transformers_[1][2]
    cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
    return cat_names + list(num_cols)


def build_shap_explainer(model, background_data, model_name: str):
    """Pick the right SHAP explainer for the model family. Tree ensembles get
    the fast, exact TreeExplainer; everything else (logistic regression) gets
    the model-agnostic Explainer with a small background sample."""
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model)
    background = shap.sample(background_data, min(100, background_data.shape[0]), random_state=42)
    return shap.Explainer(model.predict_proba, background)


def main(data_path: str, sample_size: int = 500) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODELS_DIR / "best_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model_metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    model_name = model_metadata["best_model_name"]

    _, X_test, _, y_test, _ = load_and_prepare(data_path)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_encoded_feature_names(preprocessor)

    n = min(sample_size, X_test_t.shape[0])
    sample_idx = np.random.RandomState(42).choice(X_test_t.shape[0], n, replace=False)
    X_sample = X_test_t[sample_idx]

    print(f"[explain] Building SHAP explainer for '{model_name}' on {n} test rows ...")
    explainer = build_shap_explainer(model, X_test_t, model_name)
    shap_values = explainer(X_sample)

    # For binary classifiers some SHAP explainers return a 3D array
    # (samples, features, classes) — keep the "churn" (class 1) slice.
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    # --- Bar chart: mean |SHAP value| per feature ----------------------------
    mean_abs_shap = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    top_k = 20
    ax.barh(
        [feature_names[i] for i in order[:top_k]][::-1],
        mean_abs_shap[order[:top_k]][::-1],
        color="#2E86AB",
    )
    ax.set_xlabel("Mean |SHAP value| (average impact on churn probability)")
    ax.set_title(f"Global Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=150)
    plt.close(fig)

    # --- Beeswarm summary plot (SHAP's own renderer) -------------------------
    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=150)
    plt.close(fig)

    top_features = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in order[:top_k]
    ]
    (REPORTS_DIR / "shap_top_features.json").write_text(json.dumps(top_features, indent=2))

    print(f"[explain] Saved SHAP plots to {FIGURES_DIR}/")
    print("[explain] Top 10 churn drivers:")
    for row in top_features[:10]:
        print(f"    {row['feature']:<35} {row['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate global SHAP explainability report")
    parser.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    main(args.data, sample_size=args.sample_size)
