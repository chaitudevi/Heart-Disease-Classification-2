import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific engineered features.
    """
    logger.info("Creating engineered features")

    df = df.copy()

    # Example engineered features (Heart Disease dataset)
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
            ("scaler", StandardScaler())
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
            ("encoder", 
             pd.get_dummies  # used via wrapper below
            )
        ]
    )



def build_feature_transformer(
    numeric_cols: list,
    categorical_cols: list
) -> ColumnTransformer:
    """
    Column-wise feature engineering transformer.
    """
    logger.info("Creating ColumnTransformer")

    return ColumnTransformer(
        transformers=[
            ("num", get_numeric_pipeline(), numeric_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical_cols)
        ],
        remainder="drop"
    )


def feature_engineering_pipeline(
    df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    logger.info("Starting feature engineering pipeline")

    # Step 1: Feature creation
    df = create_features(df)

    # Step 2: Column-wise transformations
    transformer = build_feature_transformer(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    transformed_array = transformer.fit_transform(df)

    # Step 3: Reconstruct DataFrame
    num_features = numeric_cols
    cat_features = categorical_cols

    feature_names = num_features + cat_features

    df_transformed = pd.DataFrame(
        transformed_array,
        columns=feature_names
    )

    # Step 4: One-hot encode categoricals
    df_transformed = pd.get_dummies(
        df_transformed,
        columns=categorical_cols,
        drop_first=True
    )

    logger.info("Feature engineering completed successfully")

    return df_transformed
