"""
Tolerant schema mapping for user-uploaded CSVs.

The model needs 19 specific features, so a CSV of unrelated data can never be
scored — that part is a hard constraint, not a bug. What this module does is
accept *any reasonable representation* of those 19 features:

  - column names in any casing/spacing/separator style
    ("MonthlyCharges", "monthly_charges", "Monthly Charges", "MONTHLY-CHARGES")
  - common alternative names ("months_tenure", "customer_gender", "is_senior")
  - value synonyms ("Y"/"yes"/"TRUE"/1 -> "Yes", "M2M" -> "Month-to-month")
  - numbers written as text ("$1,234.50", "1 234,50")
  - extra columns (customerID, Churn, anything else) — ignored, not fatal
  - missing optional columns — filled with a stated default, flagged in the report
  - the classic Telco blank-TotalCharges rows — imputed from tenure x MonthlyCharges

Anything it can't resolve is reported per-column and per-row rather than
raising, so the UI can show the user exactly what to fix (or let them map the
column by hand) instead of dying on row 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# --- Canonical value sets ---------------------------------------------------

YES_NO = ["Yes", "No"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]
YES_NO_PHONE = ["Yes", "No", "No phone service"]

_TRUE_TOKENS = {"yes", "y", "true", "t", "1", "1.0"}
_FALSE_TOKENS = {"no", "n", "false", "f", "0", "0.0"}
_NO_INTERNET_TOKENS = {
    "nointernetservice", "nointernet", "nointernetsvc", "internetno",
    "na internet", "n/a internet", "no int service",
}
_NO_PHONE_TOKENS = {
    "nophoneservice", "nophone", "phoneno", "na phone", "n/a phone", "no ph service",
}


def _norm_key(text: Any) -> str:
    """Normalize a column name or value for fuzzy comparison: lowercase and
    strip everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def _norm_loose(text: Any) -> str:
    """Like _norm_key but keeps single spaces — useful for multi-word values."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).strip().lower())).strip()


@dataclass
class FieldSpec:
    name: str
    kind: str  # "categorical" | "numeric" | "binary_int"
    aliases: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None


# Aliases are matched after _norm_key, so "monthly_charges", "Monthly Charges"
# and "MONTHLYCHARGES" all collapse to the same thing automatically — the
# alias lists below only need genuinely *different* wordings.
FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        "gender", "categorical",
        aliases=["sex", "customergender", "gender_identity"],
        allowed=["Female", "Male"],
        synonyms={"Female": ["f", "female", "woman", "w"], "Male": ["m", "male", "man"]},
        default="Female",
    ),
    FieldSpec(
        "SeniorCitizen", "binary_int",
        aliases=["senior", "issenior", "seniorcitizenflag", "isseniorcitizen", "elderly", "age65plus"],
        default=0,
    ),
    FieldSpec(
        "Partner", "categorical",
        aliases=["haspartner", "partnerflag", "married", "spouse"],
        allowed=YES_NO, default="No",
    ),
    FieldSpec(
        "Dependents", "categorical",
        aliases=["hasdependents", "dependentflag", "children", "haskids"],
        allowed=YES_NO, default="No",
    ),
    FieldSpec(
        "tenure", "numeric",
        aliases=["tenuremonths", "monthstenure", "months", "customertenure", "tenureinmonths", "monthsactive"],
        default=None, minimum=0, maximum=120,
    ),
    FieldSpec(
        "PhoneService", "categorical",
        aliases=["phone", "hasphone", "phoneflag", "telephoneservice"],
        allowed=YES_NO, default="Yes",
    ),
    FieldSpec(
        "MultipleLines", "categorical",
        aliases=["multiline", "multiplelinesflag", "extralines"],
        allowed=YES_NO_PHONE, default="No",
    ),
    FieldSpec(
        "InternetService", "categorical",
        aliases=["internet", "internettype", "internetserviceprovider", "connectiontype"],
        allowed=["DSL", "Fiber optic", "No"],
        synonyms={
            "DSL": ["dsl", "adsl", "copper"],
            "Fiber optic": ["fiberoptic", "fibreoptic", "fiber", "fibre", "fttp", "ftth", "fiber optics"],
            "No": ["no", "none", "noservice", "nointernet", "n"],
        },
        default="No",
    ),
    FieldSpec("OnlineSecurity", "categorical", aliases=["security", "onlinesecurityaddon"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec("OnlineBackup", "categorical", aliases=["backup", "onlinebackupaddon"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec("DeviceProtection", "categorical", aliases=["deviceprotect", "protection", "deviceinsurance"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec("TechSupport", "categorical", aliases=["support", "technicalsupport", "premiumsupport"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec("StreamingTV", "categorical", aliases=["tv", "streamingtelevision", "tvstreaming"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec("StreamingMovies", "categorical", aliases=["movies", "streamingfilm", "streamingfilms", "moviestreaming"], allowed=YES_NO_INTERNET, default="No"),
    FieldSpec(
        "Contract", "categorical",
        aliases=["contracttype", "contractterm", "plantype", "agreement"],
        allowed=["Month-to-month", "One year", "Two year"],
        synonyms={
            "Month-to-month": ["monthtomonth", "monthly", "m2m", "mtm", "rolling", "1month", "onemonth", "month to month"],
            "One year": ["oneyear", "1year", "12month", "12months", "annual", "yearly", "1yr"],
            "Two year": ["twoyear", "2year", "24month", "24months", "biennial", "2yr"],
        },
        default="Month-to-month",
    ),
    FieldSpec(
        "PaperlessBilling", "categorical",
        aliases=["paperless", "ebilling", "electronicbilling", "digitalbilling"],
        allowed=YES_NO, default="Yes",
    ),
    FieldSpec(
        "PaymentMethod", "categorical",
        aliases=["payment", "paymenttype", "billingmethod", "paymentmode"],
        allowed=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        synonyms={
            "Electronic check": ["electroniccheck", "echeck", "echeque", "electronicchequ", "electronniccheck", "onlinecheck"],
            "Mailed check": ["mailedcheck", "mailcheck", "postalcheck", "cheque", "check", "mailedcheque"],
            "Bank transfer (automatic)": [
                "banktransfer", "banktransferautomatic", "directdebit", "dd", "ach", "autobank", "standingorder",
            ],
            "Credit card (automatic)": [
                "creditcard", "creditcardautomatic", "card", "cc", "autocard", "debitcard", "creditcardauto",
            ],
        },
        default="Electronic check",
    ),
    FieldSpec(
        "MonthlyCharges", "numeric",
        aliases=["monthlycharge", "monthlyfee", "monthlycost", "monthlyamount", "mrr", "monthlyrevenue"],
        default=None, minimum=0, maximum=1000,
    ),
    FieldSpec(
        "TotalCharges", "numeric",
        aliases=["totalcharge", "totalfee", "totalcost", "totalamount", "lifetimevalue", "totalrevenue"],
        default=None, minimum=0, maximum=100000,
    ),
]

FIELD_SPEC_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}
REQUIRED_FIELDS = [spec.name for spec in FIELD_SPECS]
# Fields with no safe default — the model output is meaningless if we guess these.
CRITICAL_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]


# --- Column mapping ---------------------------------------------------------

def build_column_mapping(columns: list[str]) -> tuple[dict[str, str | None], dict[str, str]]:
    """Map each canonical field to a source column in the uploaded CSV.

    Returns (mapping, how) where `how` records why each match was made, so the
    UI can show the user what was auto-detected vs. guessed.
    """
    normalized_sources = {_norm_key(col): col for col in columns}
    mapping: dict[str, str | None] = {}
    how: dict[str, str] = {}

    for spec in FIELD_SPECS:
        target = _norm_key(spec.name)

        # 1. Exact match on the normalized canonical name.
        if target in normalized_sources:
            mapping[spec.name] = normalized_sources[target]
            how[spec.name] = "exact"
            continue

        # 2. Known alias.
        matched = None
        for alias in spec.aliases:
            if _norm_key(alias) in normalized_sources:
                matched = normalized_sources[_norm_key(alias)]
                break
        if matched:
            mapping[spec.name] = matched
            how[spec.name] = "alias"
            continue

        # 3. Substring containment, longest source name first so that e.g.
        #    "TotalCharges" wins over "Charges" for the TotalCharges field.
        candidates = sorted(
            (norm for norm in normalized_sources if norm and (target in norm or norm in target)),
            key=len,
            reverse=True,
        )
        if candidates:
            mapping[spec.name] = normalized_sources[candidates[0]]
            how[spec.name] = "fuzzy"
            continue

        mapping[spec.name] = None
        how[spec.name] = "missing"

    # A single source column must not feed two different fields (a "charges"
    # column shouldn't satisfy both MonthlyCharges and TotalCharges).
    seen: dict[str, str] = {}
    for target_field, source in list(mapping.items()):
        if source is None:
            continue
        if source in seen and how[target_field] == "fuzzy":
            mapping[target_field] = None
            how[target_field] = "missing"
        else:
            seen.setdefault(source, target_field)

    return mapping, how


# --- Value normalization ----------------------------------------------------

def _normalize_number(value: Any) -> float | None:
    """Parse numbers written as text: '$1,234.50', '1 234,50', '78.5%'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "null", "none", "-", "?"}:
        return None

    text = re.sub(r"[^\d,.\-]", "", text)
    if not text:
        return None

    # "1.234,56" (European) vs "1,234.56" (Anglo) vs "1234,56".
    if "," in text and "." in text:
        text = text.replace(",", "") if text.rfind(".") > text.rfind(",") else text.replace(".", "").replace(",", ".")
    elif "," in text:
        parts = text.split(",")
        text = text.replace(",", ".") if len(parts) == 2 and len(parts[-1]) in (1, 2) else text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def normalize_value(field_name: str, raw: Any) -> tuple[Any, str | None]:
    """Coerce one raw cell into the canonical value for its field.

    Returns (value, error). `error` is None on success; on failure `value` is
    None and `error` explains what was wrong.
    """
    spec = FIELD_SPEC_BY_NAME[field_name]

    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or str(raw).strip() == "":
        return None, "missing value"

    if spec.kind == "numeric":
        number = _normalize_number(raw)
        if number is None:
            return None, f"could not read {raw!r} as a number"
        if spec.minimum is not None and number < spec.minimum:
            return None, f"{number} is below the minimum {spec.minimum}"
        if spec.maximum is not None and number > spec.maximum:
            return None, f"{number} is above the maximum {spec.maximum}"
        return number, None

    token = _norm_key(raw)
    loose = _norm_loose(raw)

    if spec.kind == "binary_int":
        if token in _TRUE_TOKENS:
            return 1, None
        if token in _FALSE_TOKENS:
            return 0, None
        number = _normalize_number(raw)
        if number in (0.0, 1.0):
            return int(number), None
        return None, f"could not read {raw!r} as yes/no"

    # Categorical.
    for canonical in spec.allowed:
        if token == _norm_key(canonical):
            return canonical, None

    for canonical, variants in spec.synonyms.items():
        if any(token == _norm_key(v) or loose == _norm_loose(v) for v in variants):
            return canonical, None

    if "No internet service" in spec.allowed and (token in _NO_INTERNET_TOKENS or "nointernet" in token):
        return "No internet service", None
    if "No phone service" in spec.allowed and (token in _NO_PHONE_TOKENS or "nophone" in token):
        return "No phone service", None

    if set(spec.allowed) <= {"Yes", "No", "No internet service", "No phone service"}:
        if token in _TRUE_TOKENS:
            return "Yes", None
        if token in _FALSE_TOKENS:
            return "No", None

    return None, f"{raw!r} is not one of {spec.allowed}"


# --- Consistency + imputation ----------------------------------------------

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def apply_consistency_rules(row: dict[str, Any]) -> dict[str, Any]:
    """Align rows with the category structure the model was trained on: when a
    customer has no internet, the six internet add-ons take the distinct
    'No internet service' level rather than a plain 'No' (and likewise for
    phone/MultipleLines). A CSV that just writes 'No' everywhere is still
    scoreable, but it lands in a category the model saw far less often, so
    normalizing here keeps predictions consistent with training."""
    row = dict(row)
    if row.get("InternetService") == "No":
        for col in INTERNET_ADDONS:
            if row.get(col) in ("No", "Yes", None):
                row[col] = "No internet service"
    else:
        for col in INTERNET_ADDONS:
            if row.get(col) == "No internet service":
                row[col] = "No"

    if row.get("PhoneService") == "No":
        row["MultipleLines"] = "No phone service"
    elif row.get("MultipleLines") == "No phone service":
        row["MultipleLines"] = "No"

    return row


def impute_total_charges(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The classic Telco blank-TotalCharges case (brand-new customers). If
    tenure and MonthlyCharges are present we can reconstruct it rather than
    dropping the row."""
    row = dict(row)
    if row.get("TotalCharges") is not None:
        return row, None

    tenure, monthly = row.get("tenure"), row.get("MonthlyCharges")
    if tenure is not None and monthly is not None:
        row["TotalCharges"] = float(monthly) * max(float(tenure), 1.0)
        return row, "TotalCharges was blank — estimated from tenure x MonthlyCharges"
    return row, None


# --- Top-level normalization ------------------------------------------------

@dataclass
class NormalizationReport:
    mapping: dict[str, str | None]
    how: dict[str, str]
    defaults_used: dict[str, Any]          # column absent from the file entirely
    value_fallbacks: dict[str, int]        # column present, but N cells were unreadable
    missing_critical: list[str]
    row_errors: pd.DataFrame
    notes: list[str]
    n_input_rows: int
    n_valid_rows: int

    @property
    def ok(self) -> bool:
        return not self.missing_critical and self.n_valid_rows > 0


def normalize_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, str | None] | None = None,
    fill_defaults: bool = True,
) -> tuple[pd.DataFrame, NormalizationReport]:
    """Turn an arbitrary uploaded DataFrame into one the model can score.

    Never raises on bad data — unscoreable rows are dropped and itemized in
    `report.row_errors` so the UI can show the user exactly what went wrong.
    """
    how: dict[str, str]
    if mapping is None:
        mapping, how = build_column_mapping(list(df.columns))
    else:
        _, how = build_column_mapping(list(df.columns))
        how = {f: ("manual" if mapping.get(f) else "missing") for f in mapping}

    notes: list[str] = []
    defaults_used: dict[str, Any] = {}
    missing_critical: list[str] = []

    for spec in FIELD_SPECS:
        if mapping.get(spec.name) is not None:
            continue
        if spec.name in CRITICAL_FIELDS:
            missing_critical.append(spec.name)
        elif fill_defaults:
            defaults_used[spec.name] = spec.default

    if missing_critical:
        empty_errors = pd.DataFrame(columns=["row", "field", "problem", "value"])
        return pd.DataFrame(), NormalizationReport(
            mapping=mapping, how=how, defaults_used=defaults_used, value_fallbacks={},
            missing_critical=missing_critical, row_errors=empty_errors,
            notes=notes, n_input_rows=len(df), n_valid_rows=0,
        )

    clean_rows: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    value_fallbacks: dict[str, int] = {}
    imputed_total_charges = 0

    for position, (_, source_row) in enumerate(df.iterrows()):
        row: dict[str, Any] = {}
        row_problems: list[dict[str, Any]] = []

        for spec in FIELD_SPECS:
            source_col = mapping.get(spec.name)
            if source_col is None:
                row[spec.name] = spec.default
                continue

            value, error = normalize_value(spec.name, source_row.get(source_col))
            if error is not None:
                if spec.name in CRITICAL_FIELDS:
                    row_problems.append(
                        {"row": position + 1, "field": spec.name, "problem": error,
                         "value": source_row.get(source_col)}
                    )
                    row[spec.name] = None
                else:
                    row[spec.name] = spec.default
                    value_fallbacks[spec.name] = value_fallbacks.get(spec.name, 0) + 1
            else:
                row[spec.name] = value

        row, impute_note = impute_total_charges(row)
        if impute_note:
            imputed_total_charges += 1
            row_problems = [p for p in row_problems if p["field"] != "TotalCharges"]

        # tenure is a whole number of months; the API schema types it as int.
        if row.get("tenure") is not None:
            row["tenure"] = int(round(float(row["tenure"])))

        still_missing = [f for f in CRITICAL_FIELDS if row.get(f) is None]
        if still_missing:
            for f in still_missing:
                if not any(p["field"] == f for p in row_problems):
                    row_problems.append({"row": position + 1, "field": f, "problem": "missing value", "value": None})
            error_records.extend(row_problems)
            continue

        if row_problems:
            error_records.extend(row_problems)
            continue

        clean_rows.append(apply_consistency_rules(row))

    if imputed_total_charges:
        notes.append(
            f"{imputed_total_charges} row(s) had a blank TotalCharges — estimated from tenure x MonthlyCharges."
        )
    if defaults_used:
        notes.append(
            "Column(s) not found in your file — used a default for every row: "
            + ", ".join(f"{k}={v!r}" for k, v in defaults_used.items())
        )
    if value_fallbacks:
        notes.append(
            "Column(s) present but with unreadable values in some rows — used the default for those cells: "
            + ", ".join(f"{k} ({n} row(s))" for k, n in value_fallbacks.items())
        )

    clean_df = pd.DataFrame(clean_rows, columns=REQUIRED_FIELDS) if clean_rows else pd.DataFrame(columns=REQUIRED_FIELDS)
    errors_df = (
        pd.DataFrame(error_records, columns=["row", "field", "problem", "value"])
        if error_records
        else pd.DataFrame(columns=["row", "field", "problem", "value"])
    )

    return clean_df, NormalizationReport(
        mapping=mapping, how=how, defaults_used=defaults_used, value_fallbacks=value_fallbacks,
        missing_critical=[], row_errors=errors_df, notes=notes,
        n_input_rows=len(df), n_valid_rows=len(clean_df),
    )


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe to hand to Streamlit's table renderer.

    Streamlit serializes via pyarrow, which *segfaults the interpreter* — not
    raises — on object columns holding mixed types (strings + NaN + numbers in
    one column, which is the normal state of an arbitrary uploaded CSV).
    Stringify object columns; leave real numeric ones alone so sorting works.
    """
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns[out.dtypes == "object"]:
        out[col] = out[col].astype(str)
    return out


def read_tabular_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Read a CSV/TSV/Excel upload, sniffing delimiter and encoding rather than
    assuming comma + UTF-8."""
    name = getattr(uploaded_file, "name", "uploaded file")

    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file), "Excel"

    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()

    text = None
    encoding_used = "utf-8"
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(encoding)
            encoding_used = encoding
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    first_line = text.split("\n", 1)[0]
    delimiter = max([",", ";", "\t", "|"], key=first_line.count)
    if first_line.count(delimiter) == 0:
        delimiter = ","

    from io import StringIO

    df = pd.read_csv(StringIO(text), sep=delimiter, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]

    label = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delimiter]
    return df, f"{label}-delimited, {encoding_used}"