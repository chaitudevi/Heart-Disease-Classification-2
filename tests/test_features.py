import pandas as pd
import pytest
import numpy as np
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.data.schema import MISSING_MARKERS

logger = get_logger(__name__)

from src.features.feature_pipeline import feature_engineering_pipeline


def test_feature_engineering_output_shape(sample_df):
    X = feature_engineering_pipeline(
        df=sample_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    assert X.isnull().sum().sum() == 0
