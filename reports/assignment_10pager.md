# MLOps Assignment I — End-to-End ML Model Development, CI/CD, and Production Deployment

## 1. Objective
Build a machine learning classifier for heart disease prediction and deploy it as a cloud-ready, monitored API using modern MLOps best practices.

## 2. Repository Link
- **GitHub**: [github.com/chaitudevi/Heart-Disease-Classification-2](https://github.com/chaitudevi/Heart-Disease-Classification-2)

## 3. Dataset (Heart Disease UCI)
- **Source**: UCI Machine Learning Repository
- **Data acquisition**:
  - Script: `src/data/download_data.py`
  - Raw data folder: `data/raw/`

### Screenshot(s)
- `../screenshots/04_data_download.jpg`

## 4. Data Acquisition, Cleaning, and EDA (Task 1 — 5 marks)
### 4.1 Data Cleaning & Preprocessing
- Missing value handling, encoding, transformations
- Implementation: `src/data/preprocess.py`

### 4.2 EDA
- Notebook: `notebooks/01_eda.ipynb`
- Visuals: class balance, correlation heatmap, distributions

### Screenshot(s)
- `../screenshots/05_eda_notebook.jpg`

## 5. Feature Engineering & Model Development (Task 2 — 8 marks)
### 5.1 Feature Pipeline
- Implementation: `src/features/feature_pipeline.py`

### 5.2 Models
- Logistic Regression
- Random Forest

### 5.3 Evaluation
- Cross-validation metrics: accuracy, precision, recall, ROC-AUC
- Training entrypoint: `src/models/train.py`

### Screenshot(s)
- `../screenshots/07_training_output_1.jpg`
- `../screenshots/07_training_output_2.jpg`

## 6. Experiment Tracking (Task 3 — 5 marks)
- Tool: MLflow
- Logged:
  - Parameters
  - Metrics
  - Artifacts (ROC curve, confusion matrix, reports)

### Screenshot(s)
- `../screenshots/08_mlflow_ui.jpg`

## 7. Model Packaging & Reproducibility (Task 4 — 7 marks)
- Model bundle saved at: `artifacts/model.pkl`
- Reproducible environment: `requirements.txt`
- Preprocessing/feature transformations are part of the saved sklearn `Pipeline`

### Screenshot(s)
- `../screenshots/09_model_experiments.jpg`
- `../screenshots/09_model_metric_comparison.jpg`
- `../screenshots/09_model_artifacts.jpg`

## 8. CI/CD Pipeline & Automated Testing (Task 5 — 8 marks)
- Tests under: `tests/`
- CI workflow: `.github/workflows/github_actions.yaml`
- Stages: lint, test, train

### Screenshot(s)
- `../screenshots/13_github_actions.jpg`
- `../screenshots/11_pytest_run.jpg`
- `../screenshots/14_ci_artifacts.jpg`

## 9. Model Containerization (Task 6 — 5 marks)
- Dockerfile: `Dockerfile`
- API:
  - `/predict` accepts JSON, returns prediction + confidence
  - `/metrics` exposes Prometheus metrics

### Screenshot(s)
- `../screenshots/15_docker_build.jpg`
- `../screenshots/16_docker_run.jpg`
- `../screenshots/17_api_test.jpg`

## 10. Production Deployment (Task 7 — 7 marks)
You deployed on **Docker Desktop Kubernetes** using manifests.

- Manifest: `k8s/deployment.yaml`
- Service exposure: LoadBalancer/NodePort

### Screenshot(s)
- `../screenshots/18_k8s_deployment.jpg`
- `../screenshots/19_k8s_services.jpg`
- `../screenshots/20_k8s_pods.jpg`
- `../screenshots/21_docker_desktop_1.jpg`
- `../screenshots/21_docker_desktop_2.jpg`

## 11. Monitoring & Logging (Task 8 — 3 marks)
### 11.1 API Logging
- Middleware request logging in: `src/api/app.py`

### 11.2 Metrics + Dashboards
- Metrics endpoint: `/metrics`
- Prometheus config: `monitoring/prometheus.yml`
- Grafana provisioning + dashboard:
  - `monitoring/grafana/provisioning/`
  - `monitoring/grafana/dashboards/heart-disease-api-dashboard.json`

### Screenshot(s)
- `../screenshots/22_prometheus_metrics.jpg`
- `../screenshots/23_grafana_dashboard_1.jpg`
- `../screenshots/23_grafana_dashboard_2.jpg`
- `../screenshots/24_api_logs.jpg`

## 12. System Architecture Diagram
- Draw.io XML: `reports/system_architecture.xml`

### Screenshot(s)
- `../screenshots/25_architecture_diagram.jpg`

## 13. How to Reproduce (Production-Readiness)
### 13.1 From clean environment
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`

### 13.2 Train
- `python src/models/train.py`

### 13.3 Run API locally
- `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

### 13.4 Run monitoring stack
- `docker compose -f docker-compose.monitoring.yml up -d`

## 14. Appendix: Evidence Checklist
- CI run screenshots
- Deployment screenshots
- Grafana dashboard screenshot
- MLflow runs screenshot
