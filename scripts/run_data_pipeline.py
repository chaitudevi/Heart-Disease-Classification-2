import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import yaml
from src.data.download_data import download_dataset
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_pipeline
from src.data.eda import (
    plot_class_balance,
    plot_histograms,
    plot_correlation_heatmap
)

def main():
    with open("configs/data_config.yaml") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    figures_path = config["eda"]["figures_path"]

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    if not os.path.exists(raw_path):
        download_dataset()

    df_raw = load_raw_data(raw_path)

    df_clean = preprocess_pipeline(
        df_raw,
        config["preprocessing"]["categorical_features"],
        config["preprocessing"]["numerical_features"]
    )

    df_clean.to_csv(processed_path, index=False)

    plot_class_balance(df_clean, config["schema"]["target"], figures_path)
    plot_histograms(df_clean, figures_path)
    plot_correlation_heatmap(df_clean, figures_path)

    print(" Data pipeline executed successfully")

if __name__ == "__main__":
    main()
