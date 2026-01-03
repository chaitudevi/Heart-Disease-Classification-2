from fastapi.testclient import TestClient

from src.api.app import app
import src.models.predict as predict_module


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "api_requests_total" in text
    assert "api_request_latency_seconds" in text


def test_predict_success(monkeypatch, tmp_path):
    class _DummyModel:
        def predict(self, X):
            return [0]

        def predict_proba(self, X):
            return [[0.2, 0.8]]

    sample = {
        "age": 60,
        "sex": 1,
        "cp": 3,
        "trestbps": 120,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    monkeypatch.setattr(
        predict_module,
        "get_bundle",
        lambda: {"model": _DummyModel(), "raw_feature_names": list(sample.keys())},
    )

    # Ensure cached bundle doesn't leak across tests.
    predict_module._bundle = None

    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["prediction"] in (0, 1)
    assert "confidence" in body


def test_predict_missing_model_returns_500(monkeypatch):
    # Point the API to a missing model to demonstrate error handling.
    monkeypatch.setenv("MODEL_PATH", "/tmp/definitely-missing-model.pkl")
    predict_module._bundle = None

    sample = {
        "age": 60,
        "sex": 1,
        "cp": 3,
        "trestbps": 120,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    }

    resp = client.post("/predict", json=sample)
    assert resp.status_code == 500
    assert "Model artifact" in resp.json().get("detail", "")
