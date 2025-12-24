import os
import sys
import joblib
import pandas as pd

from src.features.feature_pipeline import feature_engineering_pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(input_json: dict):
    bundle = get_bundle()
    model = bundle["model"]
    expected_features = bundle["feature_names"]

    df = pd.DataFrame([input_json])

    X_features = feature_engineering_pipeline(
        df=df, numeric_cols=numeric_cols, categorical_cols=categorical_cols
    )

    # ALIGN FEATURES (KEY FIX)
    X_features = X_features.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(X_features)

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X_features).max())

    return {"prediction": int(prediction[0]), "confidence": confidence}
