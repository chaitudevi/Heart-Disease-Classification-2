import os
import urllib.request
import pandas as pd

# Get project root directory (heart-disease-mlops)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_FILE = "heart_disease.csv"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

def download_dataset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, OUTPUT_FILE)

    urllib.request.urlretrieve(DATA_URL, output_path)
    print(f"Dataset downloaded to {output_path}")

    df = pd.read_csv(output_path, header=None, names=COLUMN_NAMES)
    df.to_csv(output_path, index=False)
    print("Column headers added successfully")

if __name__ == "__main__":
    download_dataset()
