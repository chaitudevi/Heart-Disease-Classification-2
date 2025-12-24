import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features.feature_pipeline import (build_feature_transformer,
                                           create_features,
                                           feature_engineering_pipeline,
                                           get_categorical_pipeline,
                                           get_numeric_pipeline)


def test_create_features_adds_expected_columns():
    """
    Tests the create_features function for generating correct features
    """
    df = pd.DataFrame(
        {
            "age": [60, 55],
            "thalach": [150, 160],
            "chol": [240, 250],
            "trestbps": [120, 130],
        }
    )
    engineered = create_features(df)

    # Check columns exist
    assert "age_thalach_ratio" in engineered.columns
    assert "chol_bp_product" in engineered.columns

    # Check values are computed correctly
    assert engineered.loc[0, "age_thalach_ratio"] == 60 / (150 + 1)
    assert engineered.loc[1, "chol_bp_product"] == 250 * 130


def test_get_numeric_pipeline_returns_pipeline():
    """
    Test that get_numeric_pipeline function returns a valid scikit-learn Pipeline.
    """
    pipeline = get_numeric_pipeline()
    assert isinstance(pipeline, Pipeline)
    steps = [name for name, _ in pipeline.steps]
    assert "imputer" in steps
    assert "scaler" in steps


def test_get_categorical_pipeline_returns_pipeline():
    """
    Test that get_categorical_pipeline returns a valid scikit-learn Pipeline
    """
    pipeline = get_categorical_pipeline()
    assert isinstance(pipeline, Pipeline)
    steps = [name for name, _ in pipeline.steps]
    assert "imputer" in steps
    assert any(step == "encoder" for step in steps)


def test_build_feature_transformer_structure():
    """
    Test that build_feature_transformer constructs a ColumnTransformer correctly
    """
    numeric_cols = ["age", "chol"]
    categorical_cols = ["sex"]
    transformer = build_feature_transformer(numeric_cols, categorical_cols)
    assert isinstance(transformer, ColumnTransformer)
    transformer_names = [name for name, _, _ in transformer.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_feature_engineering_pipeline_runs_end_to_end():
    """
    Integration test for the full feature_engineering_pipeline.
    """
    df = pd.DataFrame(
        {
            "age": [60, 55],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [120, 130],
            "chol": [240, 250],
            "thalach": [150, 160],
            "target": [1, 0],
        }
    )
    numeric_cols = ["age", "trestbps", "chol", "thalach"]
    categorical_cols = ["sex", "cp"]

    processed = feature_engineering_pipeline(df, numeric_cols, categorical_cols)

    # Check output is a DataFrame
    assert isinstance(processed, pd.DataFrame)

    # Target should not be dropped
    assert "target" not in processed.columns

    # Check one-hot encoding
    assert any(col.startswith("cp_") for col in processed.columns)
    assert any(col.startswith("sex_") for col in processed.columns)

    # Row count preserved
    assert processed.shape[0] == df.shape[0]
