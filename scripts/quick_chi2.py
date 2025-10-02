import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--kids", required=True, help="CSV: y or (x,y)")
ap.add_argument("--model", required=True, help="CSV with header; y is 2nd column")
ap.add_argument("--cov", required=True, help="CSV: n×n covariance (no header)")
ap.add_argument("--label", default="")
args = ap.parse_args()

# --- kids: allow 1- or 2-col, no header
K = pd.read_csv(args.kids, header=None).to_numpy(float)
if K.ndim == 1:  # rare
    Y = K.astype(float)
else:
    if K.shape[1] == 1:
        Y = K[:, 0].astype(float)
    else:
        Y = K[:, 1].astype(float)  # second column = observed y

# --- model: headered; use 2nd col as y_model
Mdf = pd.read_csv(args.model)
if Mdf.shape[1] < 2:
    raise ValueError("Model CSV must have at least two columns (x, y_model).")
M = Mdf.iloc[:, 1].to_numpy(float)

# --- covariance
C = pd.read_csv(args.cov, header=None).to_numpy(float)

# --- align lengths and drop NaNs consistently
n = min(len(Y), len(M), C.shape[0], C.shape[1])
Y = Y[:n]; M = M[:n]; C = C[:n, :n]
mask = np.isfinite(Y) & np.isfinite(M)
Y = Y[mask]; M = M[mask]; C = C[np.ix_(mask, mask)]
n = len(Y)
if n < 2:
    raise ValueError("Too few points after alignment/NaN filtering.")

# --- robust inverse (tiny ridge if needed)
eigmin = np.linalg.eigvalsh(C).min()
if eigmin <= 0:
    eps = 1e-12 * float(np.median(np.diag(C)))
    C = C + eps * np.eye(n)
Ci = np.linalg.inv(C)

r = (M - Y).reshape(-1, 1)
chi2 = float(r.T @ Ci @ r)
nu = n  # if you want nu=n-p, adjust p here
print(f"{args.label} n={n}  chi2={chi2:.6f}  chi2/nu={chi2/nu:.6f}")
