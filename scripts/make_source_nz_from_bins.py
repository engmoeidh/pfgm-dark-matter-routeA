#!/usr/bin/env python
import argparse, numpy as np, glob, sys
ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--out_txt", required=True)
a = ap.parse_args()
files = sorted(glob.glob(a.glob))
if not files: sys.exit(f"No files matched: {a.glob}")
Z=None; SUM=None
print("Using source bins:")
for f in files:
    print("  ", f)
    arr = np.loadtxt(f)
    if arr.ndim!=2 or arr.shape[1]<2:
        sys.exit(f"Unexpected format in {f} (need 2 columns: z, n(z))")
    z, n = arr[:,0].astype(float), arr[:,1].astype(float)
    if Z is None: Z, SUM = z, n
    else:
        if not np.allclose(z, Z, atol=1e-12): sys.exit(f"z-grid mismatch: {f}")
        SUM = SUM + n
I = np.trapz(SUM, Z)
if I<=0: sys.exit("n(z) integral is zero.")
SUM /= I
np.savetxt(a.out_txt, np.column_stack([Z, SUM]), fmt="%.6e")
print(f"Wrote aggregate source n(z) → {a.out_txt}  (cols: z  n(z))")
