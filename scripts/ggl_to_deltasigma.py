import argparse
import os
import sys
import pandas as pd
import numpy as np

def normalize_columns(df):
    # Make a lower-case map and standardize to [R, DeltaSigma, DeltaSigma_err]
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in colmap:
                return colmap[k]
        raise KeyError(f"Missing required columns; have {list(df.columns)}")
    Rcol = pick("r")
    DScol = pick("deltasigma","delta_sigma","ds","delta_sigma_msun_kpc2")
    Ecol = pick("deltasigma_err","delta_sigma_err","ds_err","sigma_deltasigma")
    df = df.rename(columns={Rcol:"R", DScol:"DeltaSigma", Ecol:"DeltaSigma_err"})
    return df[["R","DeltaSigma","DeltaSigma_err"]]

def clean_kids(input_csv, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)
    df = normalize_columns(df)
    # Basic checks and cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["R"] > 0].copy()
    df = df.sort_values("R").reset_index(drop=True)
    # sanity: no non-positive uncertainties
    df = df[df["DeltaSigma_err"] > 0].copy()
    if df.empty:
        raise ValueError("After cleaning, no rows remain; check input schema/values.")
    # monotonic R assert
    if not (df["R"].values[1:] > df["R"].values[:-1]).all():
        raise ValueError("R must be strictly increasing after sort/clean.")
    df.to_csv(output_csv, index=False)
    # tiny summary
    print(f"Written {output_csv} with {len(df)} rows.")
    print(f"  R range: {df['R'].min():.4g} .. {df['R'].max():.4g} Mpc")
    # crude slope estimate (linear fit in log10 space if all positive)
    try:
        good = (df["R"]>0) & (df["DeltaSigma"]>0)
        if good.sum() >= 3:
            x = np.log10(df.loc[good,"R"].values)
            y = np.log10(df.loc[good,"DeltaSigma"].values)
            a,b = np.polyfit(x,y,1)
            print(f"  approx slope dlogΔΣ/dlogR ≈ {a:.3f}")
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Clean KiDS ΔΣ bin file and write a standardized CSV.")
    ap.add_argument("--input", default="data/raw/kids/KiDS_binA.csv", help="Input KiDS bin CSV")
    ap.add_argument("--output", default="data/processed/lensing/DeltaSigma_binA_clean.csv", help="Cleaned output CSV")
    args = ap.parse_args()
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    clean_kids(args.input, args.output)

if __name__ == "__main__":
    main()
