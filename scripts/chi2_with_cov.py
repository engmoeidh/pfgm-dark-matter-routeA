import numpy as np, pandas as pd, argparse
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kids", required=True)     # R, ΔΣ, σ (optional)
    ap.add_argument("--cov",  required=True)     # covariance matrix CSV (NxN)
    ap.add_argument("--model", required=True)    # CSV with columns: R, DS_model
    args=ap.parse_args()

    d = pd.read_csv(args.kids).to_numpy(float)
    R, DS = d[:,0], d[:,1]
    M = pd.read_csv(args.model).to_numpy(float)
    # assume model on same R (or pre-interpolated)
    DSmod = M[:,1]
    C = pd.read_csv(args.cov, header=None).to_numpy(float)
    Ci = np.linalg.inv(C)
    r = (DSmod - DS)[:,None]
    r = np.asarray(r, dtype=float).reshape(-1, 1)
    chi2 = float((r.T @ Ci @ r).item())
    nu = len(R) - 1
    print(f"chi2/nu = {chi2/nu:.3f}  (nu={nu})")

if __name__=="__main__":
    main()
