import os

BASE_DIR = "heart-disease-mlops"

PROJECT_FOLDERS = [
    f"{BASE_DIR}/data/raw",
    f"{BASE_DIR}/data/processed",
    f"{BASE_DIR}/src/data",
    f"{BASE_DIR}/src/features",
    f"{BASE_DIR}/src/models",
    f"{BASE_DIR}/src/utils",
    f"{BASE_DIR}/src/api",
    f"{BASE_DIR}/tests",
    f"{BASE_DIR}/artifacts",
    f"{BASE_DIR}/reports/figures",
    f"{BASE_DIR}/configs",
    f"{BASE_DIR}/scripts",
]

PROJECT_FILES = [
    f"{BASE_DIR}/data/raw/heart_disease.csv",
    f"{BASE_DIR}/data/processed/heart_disease_clean.csv",

    f"{BASE_DIR}/src/data/load_data.py",
    f"{BASE_DIR}/src/data/preprocess.py",
    f"{BASE_DIR}/src/data/eda.py",
    f"{BASE_DIR}/src/data/schema.py",
    f"{BASE_DIR}/src/features/feature_pipeline.py",
    f"{BASE_DIR}/src/models/train.py",
    f"{BASE_DIR}/src/models/predict.py",
    f"{BASE_DIR}/src/models/model.py",
    f"{BASE_DIR}/src/utils/logger.py",
    f"{BASE_DIR}/src/api/app.py",

    f"{BASE_DIR}/tests/test_features.py",
    f"{BASE_DIR}/tests/test_data.py",
    f"{BASE_DIR}/tests/test_models.py",

    f"{BASE_DIR}/configs/data_config.yaml",

    f"{BASE_DIR}/scripts/run_data_pipeline.py",

    f"{BASE_DIR}/requirements.txt",
    f"{BASE_DIR}/README.md",
]


def create_project_structure():
    print("Creating project directories...")
    for folder in PROJECT_FOLDERS:
        os.makedirs(folder, exist_ok=True)

    print("Creating project files...")
    for file_path in PROJECT_FILES:
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass

    print("heart-disease-mlops project structure created successfully")


if __name__ == "__main__":
    create_project_structure()
