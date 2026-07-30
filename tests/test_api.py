from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
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


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"]


def test_predict_returns_valid_response():
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert isinstance(body["churn_prediction"], bool)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ("Low", "Medium", "High")
    assert len(body["top_factors"]) > 0
    for factor in body["top_factors"]:
        assert factor["direction"] in ("increases risk", "decreases risk")


def test_predict_rejects_invalid_payload():
    bad_payload = dict(SAMPLE_PAYLOAD)
    bad_payload["Contract"] = "Not a real contract type"
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_predict_rejects_missing_field():
    bad_payload = dict(SAMPLE_PAYLOAD)
    del bad_payload["tenure"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422
