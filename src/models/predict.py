import os
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")

logger = get_logger(__name__)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        logger.info("Loading model artifact from %s", model_path)
        _bundle = joblib.load(model_path)
    return _bundle


def predict(input_json: dict):
    bundle = get_bundle()
    model = bundle["model"]
    raw_feature_names = bundle.get("raw_feature_names")

    df = pd.DataFrame([input_json])

    if raw_feature_names is not None:
        df = df.reindex(columns=raw_feature_names, fill_value=np.nan)

    prediction = model.predict(df)

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.array(model.predict_proba(df)).max())

    return {"prediction": int(prediction[0]), "confidence": confidence}
