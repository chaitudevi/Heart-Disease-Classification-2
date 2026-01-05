# Heart Disease Classification - MLOps Project

This repository contains an end-to-end MLOps pipeline to predict heart disease risk using patient data.

## Table of Contents
* [Overview](#overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Testing](#testing)
* [CI / CD Pipeline](#ci-cd-pipeline)

## Overview
This project builds a machine learning classifier to predict the risk of heart health diseases based on patient health data using the UCI Heart Disease dataset. The solution incorporates :
* Data pipeline : Automated data acquisition, cleaning and preprocessing
* Feature engineering : Scaling, encoding and transforming pipelines
* Model development : Multiple classification models (logistic regression, random forest)
* Experiment tracking : MLflow integration of parameter and metric tracking.
* Automated testing : PyTest based testcases
* CI / CD : GitHub Actions pipeline to perform linting, testing and training

## Dataset
Title : Heart disease UCI dataset
Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

The dataset contains 14+ features including :
* Age, sex, blood pressure, cholestrol levels
* Chest pain type, resting ECG results
* Maximum heart rate, exercise induced angina
* Binary tearget : Presence / absence of heart disease

## Project Structure

Heart-Disease-Classification-2/
```
    ├──src/
    │   ├── api/                       # API end-points
    │   ├── data/                      # Data loading and preprocessing
    │        ├──download_data.py
    │        ├──eda.py
    │        ├──load_data.py
    │        ├──preprocess.py
    │        ├──schema.py
    │
    │    ├──features/                    # Feature engineering pipeline
    │        ├──feature_pipeline.py
    │
    │    ├──models/                      # Model building and training
    │        ├──model.py
    │        ├──predict.py
    │        ├──train.py
    │
    │    ├──utils/                       # Utility functions
    │        ├──logger.py
    │
    ├──tests/                           # Unit tests (pytest)
    │    ├──data/
    │        ├──test_download_data.py
    │        ├──test_eda_outputs.py
    │        ├──test_load_data.py
    │        ├──test_preprocess.py
    │
    │    ├──features/
    │        test_features.py
    │
    │    ├──models/
    │        ├──test_model.py
    │
    │   ├── utils/
    │        ├──test_logger.py
    │
    ├──scripts/                       # Pipeline execution scripts
    │    ├──run_data_pipeline.py
    │
    ├──reports/                       # Test reports and visualizations
    │    ├──figures/
    │    ├──tests/
    │
    ├──data/                          # Dataset storage
    │   ├── processed/
    │    ├──raw/
    │
    ├──mlruns/                        # MLflow experiment tracking logs
    │
    ├──.github/
    │   ├── workflows/
    │        ├──github_actions.yaml    # GitHub Actions CI pipeline
    │
    ├──configs/                       # Configuration files
    │   ├──data_config.yaml
    │
    ├──requirements.txt
    ├──setup.py
    ├──README.md
```

## Installation
### Prerequisites
* Python 3.10+
* Pip

### Setup

1. Clone the repository
```bash
   git clone https://github.com/YOUR_USERNAME/Heart-Disease-Classification-2.git
   cd Heart-Disease-Classification-2
```

2. Create a virtual environment (recommended)
```bash
   python -m venv venv
   source venv/bin/activate
```

3. Install the dependencies
```bash
   pip install -r requirements.txt
```

## Usage

**Run the Data Pipeline**
```bash
python scripts/run_data_pipeline.py
```

This will:
* Download the dataset (if not present)
* Perform data cleaning and preprocessing
* Generate EDA visualizations
* Prepare features for model training

**Train the Model**
```bash
python src/models/train.py
```

This will:
* Load preprocessed data
* Train Logistic Regression and Random Forest models
* Log experiments to MLflow
* Save model artifacts to `artifacts/`

## Testing

The project includes comprehensive unit tests for all major components.

**Run All Tests**
```bash
./venv/bin/python -m pytest
```

**Run Tests with HTML Report**
```bash
./venv/bin/python -m pytest --html=reports/pytest_report.html --self-contained-html
```

**Run Specific Test Modules**
```bash
# Data tests
./venv/bin/python -m pytest tests/data/

# Feature tests
./venv/bin/python -m pytest tests/features/

# Model tests
./venv/bin/python -m pytest tests/models/

# Utility tests
./venv/bin/python -m pytest tests/utils/
```

## CI / CD pipeline
The project uses GitHub Actions for continuous integration. The pipeline runs on every push and pull request.

## Pipeline Stages
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Lint   │────▶│  Test   │────▶│  Train  │
└─────────┘     └─────────┘     └─────────┘
                                (main only)
```
1. Lint

Runs flake8 on src/ and tests/
Ensures code quality and style consistency

2. Test

Installs dependencies and package
Runs all pytest unit tests
Generates HTML test report
Uploads report as artifact

3. Train (main branch only)

Executes model training pipeline
Uploads trained model artifacts

## Workflow configuration
The CI pipeline is defined in .github/workflows/github_actions.yaml

Viewing CI Results

Navigate to the Actions tab in the GitHub repository
Select a workflow run to view details
Download artifacts (test reports, trained models) from the workflow summary

## Containerization
1. Ensure a trained model exists at `artifacts/model.pkl` (run `python src/models/train.py`).
2. Build the image: `docker build -t heart-disease-api:latest .`
3. Run locally: `docker run --rm -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts heart-disease-api:latest`
4. Test prediction: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data @sample_data/sample_request.json`

## Kubernetes Deployment
- Push the image to a registry and update `image` in `k8s/deployment.yaml`.
- Deploy: `kubectl apply -f k8s/deployment.yaml`.
- Access: `kubectl get svc heart-disease-api` or `minikube service heart-disease-api --url`.

## Helm Chart
- Package is under `charts/heart-disease-api`.
- Install: `helm install heart-api charts/heart-disease-api --set image.repository=<repo> --set image.tag=<tag>`.
- Override service type/ports: `--set service.type=NodePort --set service.port=8000`.
- Uninstall: `helm uninstall heart-api`.

## Monitoring & Observability
- Request logging enabled via middleware in the FastAPI app.
- Prometheus metrics exposed at `/metrics` (request count, latency, prediction confidence histogram).
- The service manifest includes scrape annotations for Prometheus; add the service to your Prometheus scrape config.

### Checking Prometheus & Grafana logs

#### Docker / Docker Compose
If you run Prometheus/Grafana as containers (for example via `docker compose`), you can tail logs with:

```bash
docker compose logs -f prometheus grafana
```

If you are not using compose, find the container names and view logs:

```bash
docker ps --format "table {{.Names}}\t{{.Image}}" | grep -E "prometheus|grafana"
docker logs -f <prometheus_container_name>
docker logs -f <grafana_container_name>
```

#### Kubernetes / Helm
Prometheus and Grafana are typically deployed by a monitoring chart (commonly `kube-prometheus-stack`) into a namespace such as `monitoring`.

1) Find the namespace and pod names:

```bash
kubectl get pods -A | grep -E "prometheus|grafana"
```

2) Tail logs for a specific pod:

```bash
kubectl logs -n <namespace> <pod-name> -f
```

3) If the pod has multiple containers:

```bash
kubectl logs -n <namespace> <pod-name> -c <container-name> -f
```

4) If Prometheus/Grafana were deployed as a Deployment/StatefulSet (common with Helm), you can also do:

```bash
kubectl logs -n <namespace> deploy/<grafana-deployment-name> -f
kubectl logs -n <namespace> statefulset/<prometheus-statefulset-name> -f
```

If logs are empty or you suspect restarts/crashes, check recent events:

```bash
kubectl describe pod -n <namespace> <pod-name>
```

## Sample Prediction Payload
See `sample_data/sample_request.json` for a ready-to-use request body.

