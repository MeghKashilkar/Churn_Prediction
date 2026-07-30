"""Pydantic request/response schemas for the churn prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 if the customer is a senior citizen, else 0")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months the customer has stayed with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 420.75,
            }
        }
    }


class FactorContribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases risk", "decreases risk"]


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    risk_tier: Literal["Low", "Medium", "High"]
    model_name: str
    top_factors: list[FactorContribution]


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_name: str | None = None
