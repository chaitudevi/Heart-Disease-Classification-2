import os
import sys
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features.feature_pipeline import feature_engineering_pipeline
from src.models.model import build_logestic_model, build_rf_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

mlflow.set_experiment("Heart-Disease-Classification-2")

# Load data
df = pd.read_csv("data/processed/heart_disease_clean.csv")

TARGET = "target"
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

X = df.drop(columns=[TARGET])
# Convert multi-class target to binary
df[TARGET] = (df[TARGET] > 0).astype(int)

y = df[TARGET]

#  CALL FEATURE ENGINEERING HERE
X_features = feature_engineering_pipeline(
    df=X, numeric_cols=numeric_cols, categorical_cols=categorical_cols
)

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

        cv_results = cross_validate(model, X_features, y, cv=cv, scoring=scoring)

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

best_model.fit(X_features, y)

artifact = {"model": best_model, "feature_names": X_features.columns.tolist()}
os.makedirs("artifacts", exist_ok=True)
joblib.dump(artifact, "artifacts/model.pkl")

print(f"Best model selected: {best_model_name}")

y_proba = best_model.predict_proba(X_features)[:, 1]
fpr, tpr, _ = roc_curve(y, y_proba)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
# Log final model to MLflow
with mlflow.start_run(run_name="Best_Model"):
    mlflow.log_param("selected_model", best_model_name)
    mlflow.sklearn.log_model(best_model, artifact_path="model")
    mlflow.log_artifact("roc_curve.png")
