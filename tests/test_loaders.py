import os
import pandas as pd

CLEAN_PATH = "data/processed/lensing/DeltaSigma_binA_clean.csv"

def test_kids_clean_exists_and_schema():
    assert os.path.isfile(CLEAN_PATH), f"Missing cleaned KiDS file: {CLEAN_PATH}"
    df = pd.read_csv(CLEAN_PATH)
    for col in ["R","DeltaSigma","DeltaSigma_err"]:
        assert col in df.columns, f"Missing column {col}"
    assert not df.isna().any().any(), "NaNs found in cleaned ΔΣ"
    # Strictly increasing R
    r = df["R"].values
    assert (r[1:] > r[:-1]).all(), "R is not strictly increasing"
    # Uncertainties should be positive
    assert (df["DeltaSigma_err"] > 0).all(), "Non-positive uncertainties"
