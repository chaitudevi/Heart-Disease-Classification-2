import os
import pandas as pd
import tempfile

from src.data import eda

def test_eda_outputs():
    """
    Tests the EDA functions actually generate the expected output images
    """
    df = pd.DataFrame({
        "age":[60,55], "sex":[1,0], 
        "cp":[3,2], "trestbps":[120,130],
        "chol":[240,250], "fbs":[0,1], 
        "restecg":[1,0], "thalach":[150,160], 
        "exang":[0,1], "oldpeak":[2.3,1.5], 
        "slope":[2,1], "ca":[0,1], 
        "thal":[2,3], "target":[1,0] 
    })
    outdir = tempfile.mkdtemp()

    eda.plot_class_balance(df, 'target', outdir)
    eda.plot_histograms(df, outdir)
    eda.plot_correlation_heatmap(df, outdir)

    files = os.listdir(outdir)
    assert "class_balance.png" in files
    assert "feature_histograms.png" in files
    assert "correlation_heatmap.png" in files
