"""
FastAPI service for the customer churn model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs at http://localhost:8000/docs
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerInput, HealthResponse, PredictionResponse
from src.predict import ModelNotTrainedError, load_artifacts, predict_churn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for a telecom customer and explains the prediction with SHAP.",
    version="1.0.0",
)

# Wide-open CORS so the Streamlit app (deployed on a different domain) can call this API.
# Tighten allow_origins to your Streamlit Cloud URL before shipping to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_up_model() -> None:
    """Load the model once at startup instead of on the first request, so the
    first real user doesn't eat the cold-start latency."""
    try:
        load_artifacts()
        logger.info("Model artifacts loaded successfully.")
    except ModelNotTrainedError as exc:
        logger.warning("Startup without a trained model: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        artifacts = load_artifacts()
        return HealthResponse(status="ok", model_name=artifacts["model_name"])
    except ModelNotTrainedError:
        return HealthResponse(status="model_not_loaded", model_name=None)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    try:
        result = predict_churn(customer.model_dump())
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface a clean 500 instead of a raw traceback
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)


@app.get("/")
def root() -> dict:
    return {
        "service": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
