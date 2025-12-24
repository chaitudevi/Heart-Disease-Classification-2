import numpy as np
import pandas as pd

from src.data.preprocess import (clean_data, encode_categorical,
                                 impute_missing, preprocess_pipeline)


def test_clean_data_replaces_missing():
    """
    Checks for missing values and dtype
    """
    df = pd.DataFrame({"age": [60, "?"], "sex": [1, 0]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() > 0
    assert np.issubdtype(cleaned["age"].dtype, np.number)


def test_impute_missing_numeric():
    """
    Confirms imputation fills missing numeric values.
    """
    df = pd.DataFrame({"age": [60, np.nan], "sex": [1, 0]})
    imputed = impute_missing(df, ["age"])
    assert imputed["age"].isnull().sum() == 0


def test_encode_categorical_creates_dummies():
    """
    Validates dummy variable creation for categorical encoding
    """
    df = pd.DataFrame({"sex": [1, 0, 1], "cp": [1, 2, 3]})
    encoded = encode_categorical(df, ["cp"])
    assert any(col.startswith("cp_") for col in encoded.columns)


def test_preprocess_pipeline_runs():
    """
    Integration test for the full preprocessing pipeline
    """
    df = pd.DataFrame(
        {
            "age": [60, 55],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [120, 130],
            "chol": [240, 250],
            "fbs": [0, 1],
            "restecg": [1, 0],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.5],
            "slope": [2, 1],
            "ca": [0, 1],
            "thal": [2, 3],
            "target": [1, 0],
        }
    )
    processed = preprocess_pipeline(
        df,
        ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "ca"],
        ["age", "trestbps", "chol", "thalach", "oldpeak"],
    )
    assert "target" in processed.columns
    assert processed.shape[0] == 2
