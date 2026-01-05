import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific engineered features.
    """
    logger.info("Creating engineered features")

    df = df.copy()

    # Example engineered features (Heart Disease dataset)
    df["age_thalach_ratio"] = np.nan
    df["chol_bp_product"] = np.nan

    if {"age", "thalach"}.issubset(df.columns):
        df["age_thalach_ratio"] = df["age"] / (df["thalach"] + 1)

    if {"chol", "trestbps"}.issubset(df.columns):
        df["chol_bp_product"] = df["chol"] * df["trestbps"]

    return df


def get_numeric_pipeline() -> Pipeline:
    """
    Numeric feature processing pipeline.
    """
    logger.info("Building numeric feature pipeline")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_categorical_pipeline() -> Pipeline:
    """
    Categorical feature processing pipeline.
    """
    logger.info("Building categorical feature pipeline")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_feature_transformer(
    numeric_cols: list, categorical_cols: list
) -> ColumnTransformer:
    """
    Column-wise feature engineering transformer.
    """
    logger.info("Creating ColumnTransformer")

    return ColumnTransformer(
        transformers=[
            ("num", get_numeric_pipeline(), numeric_cols),
            ("cat", get_categorical_pipeline(), categorical_cols),
        ],
        remainder="drop",
    )


def build_feature_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    logger.info("Building full feature pipeline")

    numeric_cols = list(dict.fromkeys(numeric_cols + ["age_thalach_ratio", "chol_bp_product"]))

    return Pipeline(
        steps=[
            ("feature_create", FunctionTransformer(create_features, validate=False)),
            (
                "preprocess",
                build_feature_transformer(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                ),
            ),
        ]
    )


def feature_engineering_pipeline(
    df: pd.DataFrame, numeric_cols: list, categorical_cols: list
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline")

    feature_pipeline = build_feature_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    transformed_array = feature_pipeline.fit_transform(df)
    feature_names = feature_pipeline.named_steps["preprocess"].get_feature_names_out()
    feature_names = [str(name) for name in feature_names]

    df_transformed = pd.DataFrame(transformed_array, columns=feature_names)

    logger.info("Feature engineering completed successfully")

    return df_transformed
