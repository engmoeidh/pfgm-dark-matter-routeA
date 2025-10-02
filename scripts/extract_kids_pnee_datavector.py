#!/usr/bin/env python
import argparse, sys
from astropy.io import fits
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fits", required=True)
ap.add_argument("--out_full", required=True, help="CSV; full 142-vector after scale cuts")
ap.add_argument("--out_pnee", required=True, help="CSV; PneE-only (ggl) vector using your index file")
ap.add_argument("--index", required=True, help="Index file produced earlier (PneE indices, 71 lines)")
args = ap.parse_args()

hdul = fits.open(args.fits)

# Try to find a 142-element concatenated data vector robustly.
vec = None

# 1) Common pattern: BINTABLE with column like DATA_VECTOR / DATA / VECTOR / VALUE
for h in hdul:
    data = getattr(h, "data", None)
    if data is None:
        continue
    try:
        cols = getattr(data, "columns", None)
        if cols is not None and getattr(cols, "names", None) is not None:
            names = [n.upper() for n in cols.names]
            for key in ["DATA_VECTOR", "DATA", "VECTOR", "VALUE"]:
                if key in names:
                    arr = np.array(data[cols.names[names.index(key)]]).ravel().astype(float)
                    if arr.size == 142:
                        vec = arr
                        break
            if vec is not None:
                break
    except Exception:
        pass

# 2) Fallback: naked ndarray
if vec is None:
    for h in hdul:
        data = getattr(h, "data", None)
        if isinstance(data, np.ndarray) and data.ndim == 1 and data.size == 142:
            vec = data.astype(float)
            break

# 3) Fallback: a named extension
if vec is None:
    try:
        tab = hdul["theory_data_covariance"].data
        if "VALUE" in tab.columns.names:
            arr = np.array(tab["VALUE"]).ravel().astype(float)
            if arr.size == 142: vec = arr
    except Exception:
        pass

if vec is None:
    print("ERROR: Could not locate a 142-element data vector in the FITS. HDU listing:", file=sys.stderr)
    hdul.info()
    sys.exit(2)

# Write full concatenated vector
pd.DataFrame(vec).to_csv(args.out_full, header=False, index=False)

# PneE-only slice using your index file (71 lines, 0-based)
idx = np.loadtxt(args.index, dtype=int)
pnee = vec[idx]
pd.DataFrame(pnee).to_csv(args.out_pnee, header=False, index=False)

print(f"Wrote: {args.out_full} (142) and {args.out_pnee} (71)")
