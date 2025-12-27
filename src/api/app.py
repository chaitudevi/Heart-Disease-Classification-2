import time
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.models.predict import predict
from src.utils.logger import get_logger


logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency for API requests",
    ["method", "path"],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence",
    buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
)

app = FastAPI(title="Heart Disease Risk API", version="0.1.0")


class PredictRequest(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class PredictResponse(BaseModel):
    prediction: int
    confidence: Optional[float] = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start_time
        status_code = response.status_code if response else 500

        REQUEST_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)

        logger.info(
            "Handled %s %s -> %s in %.3fs",
            request.method,
            request.url.path,
            status_code,
            duration,
        )


@app.get("/")
async def root():
    return {"status": "up", "service": "heart-disease-api"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict_endpoint(input_data: PredictRequest) -> PredictResponse:
    try:
        result = predict(input_data.model_dump())
        confidence = result.get("confidence")
        if confidence is not None:
            PREDICTION_CONFIDENCE.observe(confidence)
        return PredictResponse(**result)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        logger.error("Model artifact not found: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Model artifact missing. Run training to generate artifacts/model.pkl",
        ) from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Prediction failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
