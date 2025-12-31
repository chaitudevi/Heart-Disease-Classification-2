import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

import yaml

from src.features.feature_pipeline import build_feature_pipeline
from src.models.model import build_logestic_model, build_rf_model
from src.data.download_data import download_dataset
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_pipeline

# Ensure MLflow artifacts land in a repo-local, writable path by default.
default_tracking_dir = os.environ.get(
    "MLFLOW_TRACKING_DIR", os.path.join(PROJECT_ROOT, "mlruns")
)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file://{default_tracking_dir}")

if tracking_uri.startswith("file:"):
    tracking_path = tracking_uri.replace("file://", "", 1)
    os.makedirs(tracking_path, exist_ok=True)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(
    "Heart-Disease-Classification-2"
)

# Load data (generate if missing)
processed_csv = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease_clean.csv")
if not os.path.exists(processed_csv):
    with open(os.path.join(PROJECT_ROOT, "configs", "data_config.yaml")) as f:
        config = yaml.safe_load(f)

    raw_path = os.path.join(PROJECT_ROOT, config["data"]["raw_path"])
    processed_path = os.path.join(PROJECT_ROOT, config["data"]["processed_path"])
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if not os.path.exists(raw_path):
        download_dataset()

    df_raw = load_raw_data(raw_path)
    df_clean = preprocess_pipeline(
        df_raw,
        config["preprocessing"]["categorical_features"],
        config["preprocessing"]["numerical_features"],
    )
    df_clean.to_csv(processed_path, index=False)

df = pd.read_csv(processed_csv)

TARGET = "target"
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

X = df.drop(columns=[TARGET])
# Convert multi-class target to binary
df[TARGET] = (df[TARGET] > 0).astype(int)

y = df[TARGET]

models = {
    "Logistic Regression": build_logestic_model(),
    "Random Forest": build_rf_model(),
}

# model.fit(X_features, y)

# Cross-Validation Setup

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

# Train, Evaluate & Compare

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Log model type
        mlflow.log_param("model_type", name)

        feature_pipeline = build_feature_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        model_pipeline = Pipeline(
            steps=[
                ("features", feature_pipeline),
                ("model", model),
            ]
        )

        cv_results = cross_validate(model_pipeline, X, y, cv=cv, scoring=scoring)

        # Log metrics
        for metric in scoring:
            mean_value = np.mean(cv_results[f"test_{metric}"])
            mlflow.log_metric(metric, mean_value)

        results[name] = {
            metric: np.mean(cv_results[f"test_{metric}"]) for metric in scoring
        }

# Print Results (Report-Ready)
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# Select Best Model & Save
best_model_name = max(results, key=lambda m: results[m]["roc_auc"])
best_model = models[best_model_name]

best_feature_pipeline = build_feature_pipeline(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
)
best_pipeline = Pipeline(
    steps=[
        ("features", best_feature_pipeline),
        ("model", best_model),
    ]
)
best_pipeline.fit(X, y)

artifact = {
    "model": best_pipeline,
    "raw_feature_names": X.columns.tolist(),
}
os.makedirs("artifacts", exist_ok=True)
joblib.dump(artifact, "artifacts/model.pkl")

print(f"Best model selected: {best_model_name}")

y_proba = best_pipeline.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("reports/figures/roc_curve.png")
# Log final model to MLflow
with mlflow.start_run(run_name="Best_Model"):
    mlflow.log_param("selected_model", best_model_name)
    mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
    mlflow.log_artifact("reports/figures/roc_curve.png")
    mlflow.log_artifact("artifacts/model.pkl")
