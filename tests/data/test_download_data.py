import os
import pandas as pd
import pytest
import tempfile

from src.data import download_data

def test_download_dataset(monkeypatch):
    """ Fakes download of the dataset and 
    checks if the output file has the right structure """

    tmpdir = tempfile.mkdtemp()
    fake_file = os.path.join(tmpdir, "heart_disease.csv")

    def fake_urlretrieve(url, filename):
        with open(filename, "w") as f:
            f.write("1,1,1,120,240,0,1,150,0,2.3,2,0,2,1\n")

    monkeypatch.setattr(download_data.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmpdir)

    download_data.download_dataset()
    df = pd.read_csv(fake_file)
    assert set(download_data.COLUMN_NAMES).issubset(df.columns)
    assert df.shape[0] == 1
