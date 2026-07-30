"""
Streamlit front-end for the customer churn model.

Run locally:
    streamlit run streamlit_app/app.py

By default it calls the FastAPI service at API_URL (set via Streamlit secrets
or the API_URL environment variable). If the API is unreachable it falls back
to loading the model directly in-process, so the demo still works with just
`streamlit run` and no API running.
"""
from __future__ import annotations

import os

# Must precede any pyarrow import (Streamlit pulls it in to render tables).
# pyarrow's bundled mimalloc segfaults in mi_thread_init on Streamlit's worker
# threads under the macOS python.org 3.10 framework build; the system allocator
# doesn't. ponytail: env switch, drop it if you move off Python 3.10.0.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Make `src` importable when Streamlit Cloud runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
API_URL="http://localhost:8000"
FIELD_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

RISK_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}


def call_api(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def call_local_model(payload: dict) -> dict:
    from src.predict import predict_churn

    return predict_churn(payload)


def render_prediction(result: dict) -> None:
    prob = result["churn_probability"]
    tier = result["risk_tier"]

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": f"Churn risk — {tier}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLORS[tier]},
                    "steps": [
                        {"range": [0, 40], "color": "#E8F5E9"},
                        {"range": [40, 70], "color": "#FFF8E1"},
                        {"range": [70, 100], "color": "#FFEBEE"},
                    ],
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        verdict = "likely to churn" if result["churn_prediction"] else "likely to stay"
        st.markdown(f"**Prediction:** customer is **{verdict}** (model: `{result['model_name']}`)")

    with col2:
        factors = result["top_factors"]
        df = pd.DataFrame(factors)
        df["color"] = df["direction"].map({"increases risk": "#C62828", "decreases risk": "#2E7D32"})
        fig2 = go.Figure(
            go.Bar(
                x=df["shap_value"],
                y=df["feature"],
                orientation="h",
                marker_color=df["color"],
            )
        )
        fig2.update_layout(
            title="Top factors driving this prediction (SHAP)",
            xaxis_title="SHAP value (impact on churn probability)",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)


def single_prediction_tab() -> None:
    st.subheader("Score a single customer")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", FIELD_OPTIONS["gender"])
            senior = st.checkbox("Senior citizen")
            partner = st.selectbox("Has partner", FIELD_OPTIONS["Partner"])
            dependents = st.selectbox("Has dependents", FIELD_OPTIONS["Dependents"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", FIELD_OPTIONS["PhoneService"])
            multiple_lines = st.selectbox("Multiple lines", FIELD_OPTIONS["MultipleLines"])
            internet = st.selectbox("Internet service", FIELD_OPTIONS["InternetService"])
            online_security = st.selectbox("Online security", FIELD_OPTIONS["OnlineSecurity"])
            online_backup = st.selectbox("Online backup", FIELD_OPTIONS["OnlineBackup"])
            device_protection = st.selectbox("Device protection", FIELD_OPTIONS["DeviceProtection"])
            tech_support = st.selectbox("Tech support", FIELD_OPTIONS["TechSupport"])
            streaming_tv = st.selectbox("Streaming TV", FIELD_OPTIONS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming movies", FIELD_OPTIONS["StreamingMovies"])

        with c3:
            st.markdown("**Account & billing**")
            contract = st.selectbox("Contract", FIELD_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless billing", FIELD_OPTIONS["PaperlessBilling"])
            payment = st.selectbox("Payment method", FIELD_OPTIONS["PaymentMethod"])
            monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
            total_charges = st.number_input(
                "Total charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)), step=1.0
            )

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if not submitted:
        return

    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Scoring customer ..."):
        result = call_api(payload)
        used_fallback = False
        if result is None:
            used_fallback = True
            try:
                result = call_local_model(payload)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "Couldn't reach the API and the local model fallback also failed: "
                    f"{exc}\n\nMake sure you've run `python -m src.train` at least once."
                )
                return

    if used_fallback:
        st.caption(f"⚠️ API at `{API_URL}` unreachable — used the local model instead.")

    render_prediction(result)


def _score_batch(clean_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Score normalized rows. Prefers the local vectorized path (one model call
    for the whole file); falls back to per-row API calls if no local model is
    available, e.g. a deployed app with no models/ directory."""
    try:
        from src.predict import predict_churn_batch

        return predict_churn_batch(clean_df), "local model (vectorized)"
    except Exception:  # noqa: BLE001 — fall through to the API path
        pass

    records = clean_df.to_dict(orient="records")
    progress = st.progress(0.0, text="Scoring via API ...")
    rows, failures = [], 0
    for i, record in enumerate(records):
        result = call_api(record)
        if result is None:
            failures += 1
            rows.append({"churn_probability": None, "churn_prediction": None, "risk_tier": None})
        else:
            rows.append(
                {
                    "churn_probability": result["churn_probability"],
                    "churn_prediction": result["churn_prediction"],
                    "risk_tier": result["risk_tier"],
                }
            )
        progress.progress((i + 1) / len(records), text=f"Scoring via API ... {i + 1}/{len(records)}")
    progress.empty()

    if failures == len(records):
        return None, "failed"
    return pd.DataFrame(rows, index=clean_df.index), f"API ({failures} row(s) failed)"


def _render_mapping_editor(df: pd.DataFrame, auto_mapping: dict) -> dict:
    """Let the user correct any column the auto-detector got wrong."""
    from src.schema_mapping import CRITICAL_FIELDS, REQUIRED_FIELDS

    source_options = ["(not in my file)"] + list(df.columns)
    mapping = {}

    cols = st.columns(3)
    for i, field_name in enumerate(REQUIRED_FIELDS):
        with cols[i % 3]:
            detected = auto_mapping.get(field_name)
            index = source_options.index(detected) if detected in source_options else 0
            label = f"{field_name}{' *' if field_name in CRITICAL_FIELDS else ''}"
            choice = st.selectbox(label, source_options, index=index, key=f"map_{field_name}")
            mapping[field_name] = None if choice == "(not in my file)" else choice

    st.caption("`*` = required. Everything else falls back to a default if your file doesn't have it.")
    return mapping


def batch_prediction_tab() -> None:
    from src.schema_mapping import (
        arrow_safe,
        build_column_mapping,
        normalize_dataframe,
        read_tabular_file,
    )

    st.subheader("Score a batch of customers from a file")
    st.caption(
        "Upload a CSV, TSV or Excel file. Column names and value formats are matched "
        "automatically — `monthly_charges`, `Monthly Charges` and `MONTHLYCHARGES` all work, "
        "as do `Y`/`yes`/`TRUE`/`1` for yes-no fields. Extra columns are ignored. "
        "Anything that can't be matched is shown below so you can map it by hand."
    )

    uploaded = st.file_uploader("Upload file", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if uploaded is None:
        return

    try:
        df, format_info = read_tabular_file(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read that file: {exc}")
        return

    if df.empty:
        st.warning("That file has no rows in it.")
        return

    st.success(f"Read **{len(df)} rows × {len(df.columns)} columns** ({format_info}).")
    with st.expander("Preview your file"):
        st.dataframe(arrow_safe(df.head(10)), use_container_width=True)

    auto_mapping, how = build_column_mapping(list(df.columns))
    matched = [f for f, src in auto_mapping.items() if src]
    unmatched = [f for f, src in auto_mapping.items() if not src]

    c1, c2 = st.columns(2)
    c1.metric("Columns matched automatically", f"{len(matched)}/19")
    c2.metric("Needs attention", len(unmatched))

    if unmatched:
        st.warning("Couldn't find a column for: " + ", ".join(f"`{f}`" for f in unmatched))

    fuzzy = [f for f, kind in how.items() if kind == "fuzzy"]
    if fuzzy:
        st.info(
            "Matched by similarity (worth checking): "
            + ", ".join(f"`{f}` ← `{auto_mapping[f]}`" for f in fuzzy)
        )

    expand_editor = bool(unmatched or fuzzy)
    with st.expander("Adjust column mapping", expanded=expand_editor):
        mapping = _render_mapping_editor(df, auto_mapping)

    clean_df, report = normalize_dataframe(df, mapping=mapping)

    if report.missing_critical:
        st.error(
            "Can't score this file — these required fields have no column mapped: "
            + ", ".join(f"`{f}`" for f in report.missing_critical)
            + ". Map them above, or add them to your file."
        )
        return

    for note in report.notes:
        st.info(note)

    if len(report.row_errors):
        with st.expander(f"⚠️ {report.n_input_rows - report.n_valid_rows} row(s) couldn't be read — see why"):
            st.dataframe(arrow_safe(report.row_errors), use_container_width=True)

    if report.n_valid_rows == 0:
        st.error("None of the rows could be read. Check the problems listed above.")
        return

    st.write(f"**{report.n_valid_rows} of {report.n_input_rows} rows** are ready to score.")

    if not st.button("Run batch prediction", type="primary"):
        return

    with st.spinner("Scoring ..."):
        scores, via = _score_batch(clean_df)

    if scores is None:
        st.error(
            "Scoring failed — no local model and the API is unreachable. "
            "Run `python -m src.train` or start the API."
        )
        return

    results_df = pd.concat([clean_df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    st.caption(f"Scored via {via}.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scored", len(results_df))
    m2.metric("High risk", int((results_df["risk_tier"] == "High").sum()))
    m3.metric("Medium risk", int((results_df["risk_tier"] == "Medium").sum()))
    m4.metric("Predicted to churn", int(results_df["churn_prediction"].fillna(False).sum()))

    st.dataframe(
        arrow_safe(results_df.sort_values("churn_probability", ascending=False)),
        use_container_width=True,
    )
    st.download_button(
        "Download results as CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("📉 Customer Churn Predictor")
    st.caption(
        "Predicts the probability that a telecom customer churns, and explains why "
        "using SHAP. Backed by a FastAPI service; falls back to a local model if the "
        f"API isn't reachable. Current API target: `{API_URL}`"
    )

    tab1, tab2 = st.tabs(["Single customer", "Batch (CSV)"])
    with tab1:
        single_prediction_tab()
    with tab2:
        batch_prediction_tab()


if __name__ == "__main__":
    main()