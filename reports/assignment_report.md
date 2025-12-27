# MLOps Assignment Report

## Setup & Installation
- Clone repository and install dependencies: `pip install -r requirements.txt`
- Run data pipeline: `python scripts/run_data_pipeline.py`
- Train and log models: `python src/models/train.py` (produces `artifacts/model.pkl` and MLflow runs under `mlruns/`)
- Launch API locally: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

## EDA & Modelling Choices
- EDA notebook: `notebooks/01_eda.ipynb` (class balance, correlations, distributions).
- Feature engineering: scaling numeric columns, imputers, one-hot encoding for categoricals, plus engineered ratios/products (`feature_pipeline.py`).
- Models trained: Logistic Regression and Random Forest with 3-fold stratified CV; metrics logged: accuracy, precision, recall, ROC-AUC.
- Best model selected by ROC-AUC and saved to `artifacts/model.pkl`.

## Experiment Tracking
- MLflow configured with local file backend at `mlruns/` (override via `MLFLOW_TRACKING_URI`).
- Parameters, metrics, and ROC curve artifact logged per run; final model logged under `Best_Model` run.

## Architecture Overview
```mermaid
graph TD
  A[Client / UI] --> B[FastAPI Service]
  B --> C[Model Artifact (joblib)]
  B --> D[Prometheus Metrics]
  B --> E[Logging]
  E --> F[Stdout / Aggregator]
  D --> G[Grafana / Prometheus]
```

## CI/CD & Testing
- GitHub Actions workflow: `.github/workflows/github_actions.yaml` (lint, pytest, train on main).
- Tests: `tests/` covers data download, preprocessing, feature pipeline, model creation, logger.
- HTML reports stored under `reports/tests/` when `pytest --html` is used.

## Containerization
- Dockerfile builds FastAPI server (`uvicorn`) and exposes `/predict`, `/health`, `/metrics`.
- Requires `artifacts/model.pkl` present before build (run training step); healthcheck uses `/health`.
- Sample request payload available at `sample_data/sample_request.json`.

## Deployment
- Kubernetes manifest: `k8s/deployment.yaml` (Deployment + LoadBalancer Service with Prometheus scrape annotations).
- Deploy with `kubectl apply -f k8s/deployment.yaml`; access via service EXTERNAL-IP or `minikube service heart-disease-api --url`.
- Configure image name/tag in the manifest to match pushed Docker image.

## Monitoring & Logging
- Structured logging via `src/utils/logger.get_logger` used across API and pipeline modules.
- Prometheus metrics exposed at `/metrics` (request count/latency + prediction confidence histogram).
- Add Prometheus scrape config for `heart-disease-api` service; Grafana dashboards can plot latency and prediction confidence.

## Evidence & Links
- CI/CD results: GitHub Actions under repository Actions tab (upload HTML report artifacts).
- Deployment screenshot placeholder: add image to `reports/figures/deployment.png` and reference here.
- Repository URL: <https://github.com/YOUR_USERNAME/Heart-Disease-Classification-2>
