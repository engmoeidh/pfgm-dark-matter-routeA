import io, re, zipfile
from pathlib import Path
import numpy as np, pandas as pd

BUNDLE  = Path("data/raw/SLACS_SIS_parity_baseline_bundle.zip")
OUTCSV  = Path("data/raw/slacs_compare_autogen.csv")

def read_first_table_with_thetae(zf: zipfile.ZipFile) -> pd.DataFrame|None:
    # Pick the first CSV mentioning theta/ein/Einstein in columns
    for name in zf.namelist():
        if not name.lower().endswith(".csv"): 
            continue
        raw = zf.read(name)
        for sep in (None, ",", r"\s+"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            except Exception:
                df = None
            if df is None or df.empty: 
                continue
            cols = [str(c).strip() for c in df.columns]
            hit_theta = [c for c in cols if re.search(r'(theta[_\s]*e|einstein)', c, re.I)]
            if hit_theta:
                df.columns = cols
                return df
    return None

def main():
    if not BUNDLE.is_file():
        raise SystemExit(f"Missing bundle: {BUNDLE}")
    with zipfile.ZipFile(BUNDLE, "r") as zf:
        df = read_first_table_with_thetae(zf)
    if df is None or df.empty:
        raise SystemExit("Could not find any CSV with a θ_E / Einstein column inside the bundle.")
    # Guess name and thetaE, sigma columns
    name_col  = next((c for c in df.columns if re.search(r'(lens|name|id)', c, re.I)), df.columns[0])
    theta_col = next((c for c in df.columns if re.search(r'(theta[_\s]*e|einstein)', c, re.I)), None)
    sig_col   = next((c for c in df.columns if re.search(r'(err|sig|unc)', c, re.I)), None)

    out = pd.DataFrame(dict(
        lens = df[name_col].astype(str).str.strip(),
        thetaE_GR = pd.to_numeric(df[theta_col], errors="coerce") if theta_col else np.nan,
        sigma_GR  = pd.to_numeric(df[sig_col], errors="coerce") if sig_col else np.nan,
        thetaE_PFGM = np.nan,     # fill later when available
        sigma_PFGM  = np.nan
    ))
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTCSV, index=False)
    print(f"Wrote {OUTCSV}  (N={len(out)})")

if __name__ == "__main__":
    main()
