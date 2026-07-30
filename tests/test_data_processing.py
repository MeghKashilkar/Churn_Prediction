import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_preprocessor,
    clean_data,
    get_feature_columns,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001-AAAAA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 80.0,
                "TotalCharges": "400.0",
                "Churn": "Yes",
            },
            {
                "customerID": "0002-BBBBB",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 20.0,
                "TotalCharges": " ",  # the classic blank-string edge case
                "Churn": "No",
            },
        ]
    )


def test_clean_data_drops_blank_total_charges_row():
    raw = _sample_raw_df()
    clean = clean_data(raw)
    # The tenure=0 row with a blank TotalCharges should be dropped.
    assert len(clean) == 1
    assert clean.iloc[0]["customerID"] == "0001-AAAAA"


def test_clean_data_encodes_target_and_senior_citizen():
    raw = _sample_raw_df()
    clean = clean_data(raw)
    assert clean["Churn"].iloc[0] == 1
    assert clean["SeniorCitizen"].iloc[0] == "No"


def test_clean_data_engineers_features():
    raw = _sample_raw_df()
    clean = clean_data(raw)
    row = clean.iloc[0]
    assert row["avg_monthly_spend"] == pytest.approx(400.0 / 5)
    assert row["num_add_on_services"] == 2  # OnlineBackup + StreamingTV


def test_get_feature_columns_matches_preprocessor_inputs():
    cols = get_feature_columns()
    assert set(cols) == set(CATEGORICAL_COLS) | set(NUMERIC_COLS)


def test_build_preprocessor_fits_and_transforms():
    raw = _sample_raw_df()
    clean = clean_data(raw)
    # Duplicate rows so the one-hot encoder has >1 sample per category to fit on.
    clean = pd.concat([clean, clean], ignore_index=True)
    preprocessor = build_preprocessor()
    X = clean[get_feature_columns()]
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(clean)
    assert X_t.shape[1] > len(NUMERIC_COLS)  # one-hot expands categoricals
