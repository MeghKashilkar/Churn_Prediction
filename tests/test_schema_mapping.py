import io

import pandas as pd
import pytest

from src.schema_mapping import (
    CRITICAL_FIELDS,
    REQUIRED_FIELDS,
    apply_consistency_rules,
    arrow_safe,
    build_column_mapping,
    normalize_dataframe,
    normalize_value,
    read_tabular_file,
)


class _FakeUpload:
    """Stands in for a Streamlit UploadedFile."""

    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def read(self) -> bytes:
        return self._data


CANONICAL_ROW = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 427.5,
}


# --- Column mapping ---------------------------------------------------------

@pytest.mark.parametrize(
    "source_name, expected_field",
    [
        ("MonthlyCharges", "MonthlyCharges"),
        ("monthly_charges", "MonthlyCharges"),
        ("Monthly Charges", "MonthlyCharges"),
        ("MONTHLY-CHARGES", "MonthlyCharges"),
        ("monthly_charge", "MonthlyCharges"),
        ("months_tenure", "tenure"),
        ("customer_gender", "gender"),
        ("is_senior", "SeniorCitizen"),
        ("contract_type", "Contract"),
        ("payment_type", "PaymentMethod"),
    ],
)
def test_column_names_match_regardless_of_style(source_name, expected_field):
    mapping, _ = build_column_mapping([source_name])
    assert mapping[expected_field] == source_name


def test_extra_columns_are_ignored_not_fatal():
    columns = list(CANONICAL_ROW) + ["customerID", "Churn", "internal_notes"]
    mapping, _ = build_column_mapping(columns)
    assert all(mapping[f] is not None for f in REQUIRED_FIELDS)
    assert "customerID" not in mapping.values()


def test_one_source_column_does_not_feed_two_fields():
    # A bare "charges" column shouldn't fuzzily satisfy both charge fields.
    mapping, _ = build_column_mapping(["charges"])
    filled = [f for f in ("MonthlyCharges", "TotalCharges") if mapping[f] == "charges"]
    assert len(filled) <= 1


# --- Value normalization ----------------------------------------------------

@pytest.mark.parametrize("raw", ["Yes", "yes", "Y", "y", "TRUE", "true", 1, "1"])
def test_truthy_values_become_yes(raw):
    assert normalize_value("Partner", raw) == ("Yes", None)


@pytest.mark.parametrize("raw", ["No", "no", "N", "n", "FALSE", "false", 0, "0"])
def test_falsy_values_become_no(raw):
    assert normalize_value("Partner", raw) == ("No", None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Month-to-month", "Month-to-month"), ("month to month", "Month-to-month"),
        ("M2M", "Month-to-month"), ("monthly", "Month-to-month"),
        ("One year", "One year"), ("1 year", "One year"), ("annual", "One year"),
        ("Two year", "Two year"), ("24 months", "Two year"),
    ],
)
def test_contract_synonyms(raw, expected):
    assert normalize_value("Contract", raw) == (expected, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Fiber optic", "Fiber optic"), ("Fibre", "Fiber optic"), ("fiber", "Fiber optic"),
        ("DSL", "DSL"), ("dsl", "DSL"), ("No", "No"), ("none", "No"),
    ],
)
def test_internet_service_synonyms(raw, expected):
    assert normalize_value("InternetService", raw) == (expected, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("direct debit", "Bank transfer (automatic)"),
        ("Bank transfer (automatic)", "Bank transfer (automatic)"),
        ("credit card", "Credit card (automatic)"),
        ("echeck", "Electronic check"),
        ("Electronic check", "Electronic check"),
        ("cheque", "Mailed check"),
    ],
)
def test_payment_method_synonyms(raw, expected):
    assert normalize_value("PaymentMethod", raw) == (expected, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (85.5, 85.5), ("85.5", 85.5), ("$85.50", 85.5),
        ("85,50", 85.5),        # European decimal comma
        ("  85.50  ", 85.5),    # stray whitespace
    ],
)
def test_numbers_written_as_text(raw, expected):
    value, error = normalize_value("MonthlyCharges", raw)
    assert error is None
    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,234.50", 1234.5),   # Anglo thousands separator
        ("1.234,50", 1234.5),   # European thousands separator
        ("£1,234.50", 1234.5),  # currency symbol
        ("2050.20", 2050.2),
    ],
)
def test_thousands_separators_on_large_amounts(raw, expected):
    # Tested on TotalCharges rather than MonthlyCharges: a four-figure
    # *monthly* charge is correctly rejected by that field's range check.
    value, error = normalize_value("TotalCharges", raw)
    assert error is None
    assert value == pytest.approx(expected)


def test_implausible_monthly_charge_is_rejected_by_range_check():
    value, error = normalize_value("MonthlyCharges", "1,234.50")
    assert value is None
    assert "maximum" in error


def test_unreadable_number_reports_error_rather_than_raising():
    value, error = normalize_value("tenure", "abc")
    assert value is None
    assert "number" in error


def test_out_of_range_number_is_rejected():
    value, error = normalize_value("tenure", 9999)
    assert value is None
    assert "maximum" in error


def test_unknown_categorical_reports_allowed_values():
    value, error = normalize_value("Contract", "WEIRD")
    assert value is None
    assert "Month-to-month" in error


# --- Consistency rules ------------------------------------------------------

def test_no_internet_forces_no_internet_service_level():
    row = apply_consistency_rules({**CANONICAL_ROW, "InternetService": "No", "OnlineSecurity": "No"})
    assert row["OnlineSecurity"] == "No internet service"


def test_no_phone_forces_no_phone_service_level():
    row = apply_consistency_rules({**CANONICAL_ROW, "PhoneService": "No", "MultipleLines": "No"})
    assert row["MultipleLines"] == "No phone service"


def test_having_internet_clears_stale_no_internet_service_values():
    row = apply_consistency_rules(
        {**CANONICAL_ROW, "InternetService": "DSL", "OnlineSecurity": "No internet service"}
    )
    assert row["OnlineSecurity"] == "No"


# --- End-to-end normalization ----------------------------------------------

def test_canonical_dataframe_passes_through_unchanged():
    clean, report = normalize_dataframe(pd.DataFrame([CANONICAL_ROW]))
    assert report.n_valid_rows == 1
    assert report.ok
    assert clean.iloc[0]["MonthlyCharges"] == 85.5


def test_renamed_and_recoded_file_is_normalized():
    df = pd.DataFrame([{
        "customer_gender": "F", "is_senior": "TRUE", "married": "Y", "has_kids": "N",
        "months_tenure": 5, "phone": "Y", "multiline": "N", "internet_type": "Fibre",
        "security": "N", "backup": "N", "protection": "N", "support": "N",
        "tv": "Y", "movies": "N", "contract_type": "M2M", "paperless": "Y",
        "payment_type": "direct debit", "monthly_charge": "$85.50", "total_charge": "1,234.50",
    }])
    clean, report = normalize_dataframe(df)
    assert report.n_valid_rows == 1
    row = clean.iloc[0]
    assert row["gender"] == "Female"
    assert row["SeniorCitizen"] == 1
    assert row["InternetService"] == "Fiber optic"
    assert row["Contract"] == "Month-to-month"
    assert row["PaymentMethod"] == "Bank transfer (automatic)"
    assert row["MonthlyCharges"] == pytest.approx(85.5)
    assert row["TotalCharges"] == pytest.approx(1234.5)


def test_blank_total_charges_is_imputed_not_dropped():
    df = pd.DataFrame([{**CANONICAL_ROW, "tenure": 4, "MonthlyCharges": 50.0, "TotalCharges": " "}])
    clean, report = normalize_dataframe(df)
    assert report.n_valid_rows == 1
    assert clean.iloc[0]["TotalCharges"] == pytest.approx(200.0)
    assert any("TotalCharges" in note for note in report.notes)


def test_bad_row_is_isolated_and_good_rows_still_score():
    df = pd.DataFrame([CANONICAL_ROW, {**CANONICAL_ROW, "tenure": "abc"}, CANONICAL_ROW])
    clean, report = normalize_dataframe(df)
    assert report.n_valid_rows == 2
    assert len(report.row_errors) == 1
    assert report.row_errors.iloc[0]["field"] == "tenure"


def test_missing_optional_column_uses_default_and_is_reported():
    row = {k: v for k, v in CANONICAL_ROW.items() if k != "StreamingTV"}
    clean, report = normalize_dataframe(pd.DataFrame([row]))
    assert report.n_valid_rows == 1
    assert "StreamingTV" in report.defaults_used


def test_missing_critical_column_is_a_clear_failure_not_a_crash():
    row = {k: v for k, v in CANONICAL_ROW.items() if k != "tenure"}
    clean, report = normalize_dataframe(pd.DataFrame([row]))
    assert not report.ok
    assert "tenure" in report.missing_critical
    assert clean.empty


def test_bad_value_in_optional_column_is_a_fallback_not_a_missing_column():
    df = pd.DataFrame([{**CANONICAL_ROW, "Contract": "NONSENSE"}])
    clean, report = normalize_dataframe(df)
    assert report.n_valid_rows == 1
    assert "Contract" in report.value_fallbacks
    assert "Contract" not in report.defaults_used


def test_tenure_is_integer_for_the_api_schema():
    clean, _ = normalize_dataframe(pd.DataFrame([{**CANONICAL_ROW, "tenure": 5.0}]))
    assert isinstance(clean.iloc[0]["tenure"], (int, __import__("numpy").integer))


def test_empty_dataframe_does_not_crash():
    clean, report = normalize_dataframe(pd.DataFrame(columns=list(CANONICAL_ROW)))
    assert report.n_valid_rows == 0
    assert not clean.isna().any().any()


# --- File reading -----------------------------------------------------------

def test_reads_comma_csv():
    csv = pd.DataFrame([CANONICAL_ROW]).to_csv(index=False).encode()
    df, info = read_tabular_file(_FakeUpload(csv, "x.csv"))
    assert len(df) == 1
    assert "comma" in info


def test_reads_semicolon_csv():
    csv = pd.DataFrame([CANONICAL_ROW]).to_csv(index=False, sep=";").encode()
    df, info = read_tabular_file(_FakeUpload(csv, "x.csv"))
    assert len(df) == 1
    assert "semicolon" in info


def test_reads_tab_separated():
    tsv = pd.DataFrame([CANONICAL_ROW]).to_csv(index=False, sep="\t").encode()
    df, info = read_tabular_file(_FakeUpload(tsv, "x.tsv"))
    assert len(df) == 1
    assert "tab" in info


def test_reads_utf8_bom_without_corrupting_first_header():
    csv = pd.DataFrame([CANONICAL_ROW]).to_csv(index=False).encode("utf-8-sig")
    df, _ = read_tabular_file(_FakeUpload(csv, "x.csv"))
    assert "gender" in df.columns


# --- Arrow safety (pyarrow segfaults on mixed object columns) ---------------

def test_arrow_safe_stringifies_mixed_type_columns():
    import numpy as np

    df = pd.DataFrame({"mixed": ["text", 5, np.nan, None, 3.7]})
    out = arrow_safe(df)
    assert out["mixed"].map(type).eq(str).all()


def test_arrow_safe_leaves_numeric_columns_sortable():
    df = pd.DataFrame({"prob": [0.9, 0.1, 0.5]})
    out = arrow_safe(df)
    assert out["prob"].dtype.kind == "f"
    assert list(out.sort_values("prob")["prob"]) == [0.1, 0.5, 0.9]


def test_arrow_safe_handles_non_string_column_names():
    df = pd.DataFrame([[1, 2]], columns=[0, 1])
    assert list(arrow_safe(df).columns) == ["0", "1"]


def test_header_whitespace_is_stripped():
    csv = b" gender , tenure \nFemale,5\n"
    df, _ = read_tabular_file(_FakeUpload(csv, "x.csv"))
    assert list(df.columns) == ["gender", "tenure"]
