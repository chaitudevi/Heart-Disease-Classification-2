import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.schema import MISSING_MARKERS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Replacing custom missing markers with NaN")
    df = df.replace(MISSING_MARKERS, np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    if (df.astype(str) == "?").any().any():
        raise ValueError("'?' still present after cleaning!")
    return df


def impute_missing(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    logger.info("Imputing missing values (median strategy)")
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    logger.info("Encoding categorical features (one-hot)")
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def preprocess_pipeline(
    df: pd.DataFrame, categorical_cols: list, numeric_cols: list
) -> pd.DataFrame:
    df = clean_data(df)
    df = impute_missing(df, numeric_cols)
    # df = encode_categorical(df, categorical_cols)
    logger.info("Preprocessing completed")
    return df
