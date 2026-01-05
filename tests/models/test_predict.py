import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.feature_pipeline import build_feature_pipeline
from src.models import predict as predict_module


def test_predict_uses_saved_pipeline(tmp_path, monkeypatch):
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    X = pd.DataFrame(
        {
            "age": [52, 60],
            "sex": [1, 0],
            "cp": [0, 3],
            "trestbps": [125, 120],
            "chol": [212, 240],
            "fbs": [0, 1],
            "restecg": [1, 0],
            "thalach": [168, 150],
            "exang": [0, 1],
            "oldpeak": [1.0, 2.3],
            "slope": [2, 2],
            "ca": [0, 0],
            "thal": [2, 2],
        }
    )
    y = [0, 1]

    feature_pipeline = build_feature_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    pipeline = Pipeline(
        steps=[
            ("features", feature_pipeline),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)

    artifact_path = tmp_path / "model.pkl"
    joblib.dump(
        {
            "model": pipeline,
            "raw_feature_names": X.columns.tolist(),
        },
        artifact_path,
    )

    monkeypatch.setenv("MODEL_PATH", str(artifact_path))
    predict_module._bundle = None

    result = predict_module.predict(
        {
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "thal": 2,
        }
    )

    assert "prediction" in result
    assert result["prediction"] in (0, 1)

    assert "confidence" in result
    assert result["confidence"] is None or 0.0 <= result["confidence"] <= 1.0
