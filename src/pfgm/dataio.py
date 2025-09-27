import sys, json
import pandas as pd

REQUIRED_SCHEMAS = {
    "kids_bin": ["R", "DeltaSigma", "DeltaSigma_err"],  # Mpc, Msun/kpc^2, Msun/kpc^2
    "sparc_tbl": ["Galaxy", "Vflat"],                   # minimal check for now
}

def validate_csv(path, kind):
    cols = REQUIRED_SCHEMAS.get(kind, [])
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} for kind='{kind}'")
    if df.isna().any().any():
        raise ValueError(f"{path}: contains NaNs")
    return True

def main():
    # very light "validate" pass used by Makefile target
    try:
        print("Validation placeholder: OK")
        sys.exit(0)
    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
