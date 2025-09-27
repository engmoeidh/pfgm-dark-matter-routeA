#!/usr/bin/env python3
import os, glob, json
import numpy as np, pandas as pd
from math import pi
try:
    from scipy import special
    def J2(x): return special.jv(2,x)
except Exception:
    from mpmath import besselj
    def J2(x):
        x = np.asarray(x)
        return np.array([float(besselj(2, float(xx))) for xx in x])

# Config
H = 0.674
OMEGA_M = 0.0493 + 0.2647
RHO_C0 = 2.775e11 * H**2  # Msun/Mpc^3
RHO_M = OMEGA_M * RHO_C0

def load_cols(df, keys):
    for k in keys:
        for c in df.columns:
            if c.lower() == k.lower():
                return df[c].values.astype(float)
    raise KeyError(f"Missing column among {keys}")

def delta_sigma_2h(Rvals, k, Pk, b0, b2, lam, p=1.0):
    mu = 1.0/(1.0 + (k*lam)**2)**p
    bias = b0 + b2*k**2
    Pgm = bias * mu * Pk
    pref = RHO_M/(2.0*np.pi)
    out = []
    for R in Rvals:
        x = k*R
        J = J2(x)
        integrand = k * Pgm * J
        I = np.trapz(integrand, k)
        DS = pref*I / 1e6  # Msun/kpc^2
        out.append(DS)
    return np.array(out)

def fit_bin(kids_path, oneh_path, pnl_csv, out_prefix, random_seed=123, N=1600, lock_A=False):
    kids = pd.read_csv(kids_path)
    oneh = pd.read_csv(oneh_path)
    pnl  = pd.read_csv(pnl_csv)
    R = load_cols(kids, ["R","R_Mpc","R_mpc"])
    DSd = load_cols(kids, ["DeltaSigma","DS","Delta_Sigma","DeltaSigma_Msun_per_kpc2"])
    eDS = load_cols(kids, ["DeltaSigma_err","DS_err","Delta_Sigma_err","Err"])
    DS1 = load_cols(oneh, ["DeltaSigma","DS","Delta_Sigma","DeltaSigma_Msun_per_kpc2"])
    k = pnl["k_1perMpc"].values.astype(float)
    P = pnl["P_Mpc3"].values.astype(float)

    w = 1.0/(eDS**2)
    def best_A_given_params(b0,b2,lam):
        DS2 = delta_sigma_2h(R, k, P, b0,b2,lam)
        if lock_A:
            A = 1.0
        else:
            num = np.sum(w * DS1 * (DSd - DS2))
            den = np.sum(w * (DS1**2))
            A = num/den
        model = A*DS1 + DS2
        chi2 = np.sum(w * (model - DSd)**2)
        return A, chi2, model, DS2

    rng = np.random.default_rng(random_seed)
    b0_s = rng.uniform(1.0, 3.0, N)
    b2_s = rng.uniform(0.0, 10.0, N)
    lam_s= rng.uniform(0.0, 3.0, N)

    best = None
    for b0, b2, lam in zip(b0_s, b2_s, lam_s):
        A, chi2, model, DS2 = best_A_given_params(b0,b2,lam)
        if (best is None) or (chi2 < best[0]):
            best = (chi2, A, b0, b2, lam, model, DS2)
    chi2, A, b0, b2, lam, model, DS2 = best

    nu = len(R) - (3 if lock_A else 4)
    meta = {
        "kids": os.path.basename(kids_path),
        "onehalo": os.path.basename(oneh_path),
        "pnl": os.path.basename(pnl_csv),
        "best": {
            "chi2": float(chi2),
            "chi2_over_nu": float(chi2/max(nu,1)),
            "nu": int(nu),
            "A_1h": float(A),
            "b0": float(b0),
            "b2_Mpc2": float(b2),
            "lambda_Mpc": float(lam)
        }
    }
    pd.DataFrame({"R_Mpc":R, "DeltaSigma_2h_Msun_per_kpc2":DS2}).to_csv(out_prefix+"_2halo.csv", index=False)
    pd.DataFrame({"R_Mpc":R, "DeltaSigma_total":model}).to_csv(out_prefix+"_total.csv", index=False)
    with open(out_prefix+"_fit.json","w") as f:
        json.dump(meta,f,indent=2)
    return meta

def main():
    pnl_csv = "Pnl_z0p286_HMcode.csv"  # expected filename from CLASS+converter
    if not os.path.exists(pnl_csv):
        print("Missing", pnl_csv, "in current directory. Place the HMcode CSV here.")
        return
    # Find KiDS bins
    kids_files = sorted(glob.glob("KiDS_bin*.csv"))
    if not kids_files:
        print("No KiDS_bin*.csv files found in current directory.")
        return
    # Try to match corresponding 1-halo files by naming pattern
    results = []
    for kids_path in kids_files:
        tag = os.path.splitext(os.path.basename(kids_path))[0].replace("KiDS_","")
        oneh_candidates = [
            f"routeA_m0p18_{tag}.csv",
            f"routeA_m0p18_{tag}_1halo.csv",
            f"routeA_m0p18_{tag.replace('bin','binA')}.csv"
        ]
        oneh_path = None
        for c in oneh_candidates:
            if os.path.exists(c):
                oneh_path = c; break
        if oneh_path is None:
            print("Missing 1-halo file for", kids_path, "—skipping.")
            continue
        meta_free  = fit_bin(kids_path, oneh_path, pnl_csv, out_prefix=f"HMfit_{tag}_freeA", lock_A=False)
        meta_lock  = fit_bin(kids_path, oneh_path, pnl_csv, out_prefix=f"HMfit_{tag}_A1", lock_A=True)
        results.append({"tag":tag, "freeA":meta_free["best"], "A1":meta_lock["best"]})
        print("Finished", tag, "→ λ_freeA={:.3f} Mpc, λ_A1={:.3f} Mpc".format(
            meta_free["best"]["lambda_Mpc"], meta_lock["best"]["lambda_Mpc"]
        ))
    # Summarize
    if results:
        rows = []
        for r in results:
            rows.append([r["tag"], r["freeA"]["lambda_Mpc"], r["A1"]["lambda_Mpc"],
                         r["freeA"]["chi2_over_nu"], r["A1"]["chi2_over_nu"]])
        df = pd.DataFrame(rows, columns=["bin","lambda_freeA_Mpc","lambda_A1_Mpc","chi2nu_freeA","chi2nu_A1"])
        df.to_csv("HMfit_lambda_summary.csv", index=False)
        print("Wrote HMfit_lambda_summary.csv")

if __name__ == "__main__":
    main()
