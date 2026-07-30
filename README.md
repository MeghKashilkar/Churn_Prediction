# Customer Churn Prediction System

End-to-end churn prediction project: data cleaning & feature engineering,
comparison of four classifiers, SHAP explainability (global + per-prediction),
a FastAPI service, and a Streamlit app for interactive scoring.

Built as portfolio project #1 for a UK data science job search.

**🔗 Live demo:** [churn-prediction-megh.streamlit.app](https://churn-prediction-megh.streamlit.app)

## How to use the live app

1. Open [churn-prediction-megh.streamlit.app](https://churn-prediction-megh.streamlit.app).
2. **Single Customer** tab - fill in the form (contract type, tenure, monthly
   charges, services, etc.) and submit. You'll get a churn probability gauge
   plus a SHAP bar chart showing which fields pushed the prediction up or down.
3. **Batch** tab - upload a CSV/Excel file of customers and get a churn score
   for every row, downloadable as a results file. Column names don't need to
   match exactly (`monthly_charges`, `Monthly Charges`, etc. all work) - see
   [Batch upload - accepted formats](#batch-upload--accepted-formats) below
   for what's auto-detected vs. what you'll be asked to map by hand.

The app calls a FastAPI backend on Render; if that backend is asleep (free
tier spins down after 15 min idle) the first request may take 30-60s to wake
it, or the app falls back to scoring locally.

## Why this project

Telecom churn is a classic but well-scoped business problem: predict which
customers are about to leave so retention teams can intervene. It's a good
first portfolio project because the dataset is clean enough to focus on
modeling and deployment craft rather than data wrangling, but still has real
class imbalance (~27% churn) and a mix of categorical/numeric features that
make explainability genuinely useful rather than decorative.

## Architecture

```
Raw CSV --> src/data_processing.py --> src/train.py --> models/*.pkl
                                              |
                                              v
                                        src/explain.py (SHAP report)
                                              |
                  +---------------------------+---------------------------+
                  v                                                       v
         api/main.py (FastAPI)                              streamlit_app/app.py
         POST /predict, GET /health                          calls the API;
         deployed on Render                                  falls back to a local
                                                               model if API is down
                                                               deployed on Streamlit
                                                               Community Cloud
```

`src/predict.py` is shared by both the API and the Streamlit app so
prediction + explanation logic can't drift between the two surfaces.

## Repo layout

```
churn-prediction/
├── data/raw/                  # raw CSV goes here (see "Get the data" below)
├── data/processed/            # (optional) cached cleaned data
├── src/
│   ├── data_processing.py     # cleaning, feature engineering, preprocessing pipeline
│   ├── train.py                # trains + compares 4 models, saves the best one
│   ├── explain.py              # global SHAP report (bar + beeswarm plots)
│   └── predict.py              # shared inference + local-SHAP logic (used by API & app)
├── api/
│   ├── main.py                 # FastAPI app (/predict, /health)
│   ├── schemas.py               # Pydantic request/response models
│   └── Dockerfile
├── streamlit_app/
│   └── app.py                   # single + batch prediction UI
├── tests/
│   ├── conftest.py              # bootstraps a tiny synthetic model if none exists
│   ├── test_data_processing.py
│   └── test_api.py
├── models/                      # trained model + preprocessor land here (gitignored)
├── reports/                     # model comparison table + SHAP/ROC figures
├── requirements.txt
├── render.yaml                  # Render blueprint for the API
└── Procfile                     # alternative start command (Render/Heroku-style)
```

## Get the data

This repo ships with `data/raw/Telco-Customer-Churn-sample.csv` - **487 rows**
sampled from the classic IBM/Kaggle "Telco Customer Churn" dataset, included
purely so the pipeline runs out of the box for a smoke test. It's not enough
data for a credible portfolio result.

Before training for real, download the full 7,043-row dataset (one command,
takes a few seconds):

```bash
curl -o data/raw/Telco-Customer-Churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

or grab it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(same schema, needs a Kaggle account). Either way it lands at
`data/raw/Telco-Customer-Churn.csv`, which is the default path every script
below expects - pass `--data data/raw/Telco-Customer-Churn-sample.csv`
instead if you just want to smoke-test the code first.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on how this repo was built:** the code in this project was written
> and reviewed but not executed in the environment it was generated in
> (no outbound package installs were possible there). Run the smoke test
> below first - if anything errors, paste the traceback back and it'll get
> fixed fast.

### Smoke test

```bash
python -m src.data_processing --data data/raw/Telco-Customer-Churn-sample.csv
pytest -q
```

## Train

```bash
python -m src.train --data data/raw/Telco-Customer-Churn.csv
```

This compares **logistic regression, random forest, gradient boosting, and
XGBoost** with 5-fold stratified cross-validation, evaluates all four on a
held-out 20% test set (accuracy, precision, recall, F1, ROC-AUC), and saves:

- `models/best_model.pkl`, `models/preprocessor.pkl` - the winning pipeline
- `reports/model_comparison.csv` - every model's metrics side by side
- `reports/figures/roc_curves.png`, `reports/figures/confusion_matrix.png`

**Fill in your real numbers here after running it** (this is exactly the kind
of quantified bullet your CV needs):

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| _run `python -m src.train` and paste `reports/model_comparison.csv` here_ | | | | |

## Explain

```bash
python -m src.explain --data data/raw/Telco-Customer-Churn.csv
```

Produces `reports/figures/shap_summary_bar.png` and `shap_summary_beeswarm.png`,
global feature importance for the winning model, plus
`reports/shap_top_features.json`. Contract type, tenure, and internet service
are the usual top drivers on this dataset; your actual ranking will be in the
JSON.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 420.75
      }'
```

## Run the Streamlit app

```bash
export API_URL=http://localhost:8000   # optional - defaults to this anyway
streamlit run streamlit_app/app.py
```

Two tabs: score one customer via a form (gauge chart + SHAP bar chart of what
drove the prediction), or upload a file and score a batch with a downloadable
results file.

### Batch upload - accepted formats

The batch tab doesn't require your file to match the Telco dataset's exact
column names or value spellings. `src/schema_mapping.py` normalizes:

| Variation | Examples that all work |
|---|---|
| Column naming | `MonthlyCharges`, `monthly_charges`, `Monthly Charges`, `MONTHLY-CHARGES`, `monthly_charge` |
| Alternative names | `months_tenure` → `tenure`, `customer_gender` → `gender`, `is_senior` → `SeniorCitizen`, `contract_type` → `Contract` |
| Yes/no values | `Yes`, `yes`, `Y`, `TRUE`, `1` |
| Category synonyms | `M2M`/`monthly` → `Month-to-month`; `Fibre`/`fiber` → `Fiber optic`; `direct debit` → `Bank transfer (automatic)`; `echeck` → `Electronic check` |
| Numbers as text | `$85.50`, `1,234.50`, `1.234,50` (European), `85,50` |
| File format | comma / semicolon / tab / pipe delimited, `.xlsx`, UTF-8, UTF-8-BOM, Latin-1 |
| Extra columns | `customerID`, `Churn`, anything else - ignored, not an error |
| Missing optional columns | Filled with a stated default, reported in the UI |
| Blank `TotalCharges` | Estimated from `tenure × MonthlyCharges` (the classic new-customer rows) |

Only `tenure`, `MonthlyCharges` and `TotalCharges` are strictly required,
since the model's output would be meaningless if those were guessed. Everything else
falls back to a default that's shown to you.

Whatever the auto-detector can't resolve appears in an **Adjust column
mapping** panel where you pick the source column by hand, and any row that
still can't be read is listed with the specific reason rather than failing the
whole upload. Batch scoring runs as a single vectorized model call, so a
7,000-row file scores in one pass instead of 7,000 HTTP requests.

## Tests

```bash
pytest -q
```

`tests/conftest.py` bootstraps a tiny model from synthetic data if
`models/best_model.pkl` doesn't exist yet, so the suite is self-contained,
useful for CI, but **run `src/train.py` on the real data before trusting any
numbers you put on your CV.**

## Deployment

### API → Render

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo. `render.yaml` at the
   root configures everything (Docker build from `api/Dockerfile`, free plan,
   health check on `/health`).
3. Render builds and gives you a URL like `https://churn-prediction-api.onrender.com`.
   Note it - the Streamlit app needs it.

   Free-tier Render services spin down after 15 minutes idle and take ~30-60s
   to wake back up on the next request. Mention this if you demo it live.

### Streamlit app → Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
2. Main file path: `streamlit_app/app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   API_URL = "https://churn-prediction-api.onrender.com"
   ```
   (use the real Render URL from the step above - see
   `.streamlit/secrets.toml.example`).
4. Deploy. Streamlit Cloud installs everything from the root `requirements.txt`.

   Note: `models/` is gitignored, so the deployed Streamlit app's local-model
   fallback won't have a model to load unless you either (a) commit the
   trained `.pkl` files for this one project, overriding the gitignore rule,
   or (b) rely on the API being up. Given Render free tier sleeps, committing
   the model artifacts is the more reliable option for a live demo link.

### Docker (either service, run anywhere)

```bash
docker build -f api/Dockerfile -t churn-api .
docker run -p 8000:8000 churn-api
```

## Design notes / talking points for interviews

- **Class imbalance** (~27% churn): handled via `class_weight="balanced"`
  (logistic regression, random forest) and `scale_pos_weight` (XGBoost)
  rather than resampling, so the training distribution still reflects
  reality - model comparison uses ROC-AUC and F1, not accuracy, since
  accuracy is misleading on imbalanced data.
- **No data leakage**: this dataset (unlike some of the newer IBM-enriched
  versions floating around) has no pre-computed churn-score or CLTV columns
  to accidentally leak the target - every feature is something you'd know
  about a customer *before* they churn.
- **Preprocessing is a fitted, persisted object** (`ColumnTransformer`
  saved via `joblib`), not ad-hoc code duplicated between training and
  serving - the API and Streamlit app both call the exact same
  `src/predict.py::predict_churn`, so there's no train/serve skew.
- **Local + global explainability**: `src/explain.py` gives the "what
  matters overall" story for a model card; `src/predict.py`'s per-request
  SHAP values give the "why did *this* customer score high" story a
  retention agent would actually want.
