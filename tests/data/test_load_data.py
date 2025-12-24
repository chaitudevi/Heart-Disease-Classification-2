import tempfile

import pandas as pd

from src.data.load_data import load_raw_data
from src.data.schema import EXPECTED_COLUMNS


def test_load_raw_data(tmp_path):
    """
    Validates if the csv file is correctly read ans schema preserved
    """
    file = tmp_path / "heart.csv"
    pd.DataFrame(
        [[60, 1, 3, 120, 240, 0, 1, 150, 0, 2.3, 2, 0, 2, 1]], columns=EXPECTED_COLUMNS
    ).to_csv(file, index=False)
    df = load_raw_data(file)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.shape[0] == 1
