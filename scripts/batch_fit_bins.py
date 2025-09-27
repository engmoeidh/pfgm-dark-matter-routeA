import os, json, argparse, yaml
import numpy as np
import pandas as pd
from scipy import optimize, interpolate
import matplotlib.pyplot as plt

from src.pfgm.kernels import mu_lens, hankel_j2

def load_manifest(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_kids_clean(csv_path, cov_path=None):
    df = pd.read_csv(csv_path)
    R = df["R"].values.astype(float)
    DS = df["DeltaSigma"].values.astype(float)
    S  = df["DeltaSigma_err"].values.astype(float)
    # Covariance (optional)
    C = None
    if cov_path and os.path.isfile(cov_path):
        C = np.loadtxt(cov_path, delimiter=",")
        # fallback if it has headers/unknown formatting
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != len(R):
            C = None
    return R, DS, S, C

def load_routeA_1h(csv_path, R_target):
    import pandas as pd, numpy as np
    def try_read(header):
        return pd.read_csv(csv_path, header=header)
    # Try header=None first (two columns), else header=0 with named columns
    for hdr in [None, 0]:
        try:
            df = try_read(hdr)
            # If exactly two columns, assume first is R, second is ΔΣ_1h
            if df.shape[1] == 2:
                Rtab = df.iloc[:,0].astype(float).values
                oneh = df.iloc[:,1].astype(float).values
                break
            # Otherwise try to find named columns
            cols = {c.lower().strip(): c for c in df.columns}
            Rcol = cols.get('r') or list(df.columns)[0]
            Dcol = cols.get('deltasigma_1h') or cols.get('ds_1h') or cols.get('deltasigma') or list(df.columns)[1]
            Rtab = df[Rcol].astype(float).values
            oneh = df[Dcol].astype(float).values
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Cannot parse 1-halo CSV: {csv_path}")
    # Interpolate onto requested radii
    from scipy import interpolate
    f = interpolate.InterpolatedUnivariateSpline(Rtab, oneh, k=1, ext=1)
    return f(R_target)


def build_pnl(k_file, kmin, kmax, nk):
    # load HMcode nonlinear P(k) columns: k [1/Mpc], P [Mpc^3]
    df = pd.read_csv(k_file)
    # try best-guess columns
    cols = {c.lower(): c for c in df.columns}
    kcol = cols.get("k", list(df.columns)[0])
    pcol = cols.get("p_mpc3", cols.get("p", list(df.columns)[1]))
    ktab = df[kcol].values.astype(float)
    ptab = df[pcol].values.astype(float)
    # enforce positive k-range and monotone
    mask = (ktab>0) & (ptab>0)
    kt, pt = ktab[mask], ptab[mask]
    ks = np.geomspace(max(kmin, kt.min()), min(kmax, kt.max()), nk)
    P  = np.interp(ks, kt, pt)
    return ks, P

def model_2h(R, k, Pnl, b0, b2, lam):
    # P_gm(k) = (b0 + b2 k^2) * mu_lens(k;lam) * P_nl(k)
    Pgm = (b0 + b2*(k**2)) * mu_lens(k, lam) * Pnl
    # Simple Hankel-like projection with j2
    ds2h = hankel_j2(k, Pgm, R)
    # Units: this outputs an arbitrary normalization proportional to ΔΣ_2h shape.
    # For our purposes (relative weighting vs 1h), this suffices.
    return ds2h

def chi2(model, data, cov=None, sigma=None):
    r = model - data
    if cov is not None:
        try:
            inv = np.linalg.inv(cov)
            return float(r @ inv @ r)
        except Exception:
            pass
    if sigma is None:
        raise ValueError("Need sigma (diag) if covariance is not provided.")
    return float(np.sum((r/sigma)**2))

def fit_binA(man, out_tag="binA"):
    R, DS, S, C = load_kids_clean(man["paths"]["kids_binA_clean"], man["paths"].get("kids_binA_cov"))
    oneh = load_routeA_1h(man["paths"]["routeA_1h_binA"], R)
    k, Pnl = build_pnl(man["paths"]["p_nl"], man["fitting"]["k_min"], man["fitting"]["k_max"], man["fitting"]["n_k"])

    # free-A fit
    x0 = np.array([
        man["fitting"]["initial"]["A1h"],
        man["fitting"]["initial"]["b0"],
        man["fitting"]["initial"]["b2"],
        man["fitting"]["initial"]["lam"],
    ])
    lb = np.array([man["fitting"]["bounds"]["A1h"][0], man["fitting"]["bounds"]["b0"][0],
                   man["fitting"]["bounds"]["b2"][0], man["fitting"]["bounds"]["lam"][0]])
    ub = np.array([man["fitting"]["bounds"]["A1h"][1], man["fitting"]["bounds"]["b0"][1],
                   man["fitting"]["bounds"]["b2"][1], man["fitting"]["bounds"]["lam"][1]])

    def residual_free(x):
        A1h, b0, b2, lam = x
        twoh = model_2h(R, k, Pnl, b0, b2, lam)
        mod  = A1h*oneh + twoh
        if C is not None:
            # return whitened residuals via Cholesky if possible
            try:
                L = np.linalg.cholesky(C)
                return np.linalg.solve(L, mod-DS)
            except Exception:
                pass
        return (mod-DS)/S

    res = optimize.least_squares(residual_free, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    A1h_f, b0_f, b2_f, lam_f = res.x
    twoh_f = model_2h(R, k, Pnl, b0_f, b2_f, lam_f)
    mod_f  = A1h_f*oneh + twoh_f
    chi2_f = chi2(mod_f, DS, cov=C, sigma=S)
    nu_f   = len(R) - len(res.x)
    out_free = {
        "bin": "A",
        "mode": "freeA",
        "A1h": float(A1h_f), "b0": float(b0_f), "b2": float(b2_f), "lam": float(lam_f),
        "chi2": float(chi2_f), "nu": int(nu_f), "chi2/nu": float(chi2_f/max(nu_f,1))
    }

    # A1h = 1 (SPARC-locked) fit
    def residual_locked(y):
        b0, b2, lam = y
        twoh = model_2h(R, k, Pnl, b0, b2, lam)
        mod  = 1.0*oneh + twoh
        if C is not None:
            try:
                L = np.linalg.cholesky(C)
                return np.linalg.solve(L, mod-DS)
            except Exception:
                pass
        return (mod-DS)/S

    y0 = np.array([man["fitting"]["initial"]["b0"], man["fitting"]["initial"]["b2"], man["fitting"]["initial"]["lam"]])
    lb2 = np.array([man["fitting"]["bounds"]["b0"][0], man["fitting"]["bounds"]["b2"][0], man["fitting"]["bounds"]["lam"][0]])
    ub2 = np.array([man["fitting"]["bounds"]["b0"][1], man["fitting"]["bounds"]["b2"][1], man["fitting"]["bounds"]["lam"][1]])

    res2 = optimize.least_squares(residual_locked, y0, bounds=(lb2, ub2), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    b0_l, b2_l, lam_l = res2.x
    twoh_l = model_2h(R, k, Pnl, b0_l, b2_l, lam_l)
    mod_l  = 1.0*oneh + twoh_l
    chi2_l = chi2(mod_l, DS, cov=C, sigma=S)
    nu_l   = len(R) - len(res2.x)
    out_lock = {
        "bin": "A",
        "mode": "A1",
        "A1h": 1.0, "b0": float(b0_l), "b2": float(b2_l), "lam": float(lam_l),
        "chi2": float(chi2_l), "nu": int(nu_l), "chi2/nu": float(chi2_l/max(nu_l,1))
    }

    # Save outputs
    os.makedirs("results/tables", exist_ok=True)
    with open(f"results/tables/{out_tag}_fit_freeA.json","w") as f:
        json.dump(out_free, f, indent=2)
    with open(f"results/tables/{out_tag}_fit_A1.json","w") as f:
        json.dump(out_lock, f, indent=2)

    # Plots
    os.makedirs("figures/lensing", exist_ok=True)
    for tag, mod, twoh, A1h in [
        ("freeA", mod_f, twoh_f, A1h_f),
        ("A1",   mod_l, twoh_l, 1.0),
    ]:
        plt.figure(figsize=(6.2,4.6))
        plt.errorbar(R, DS, yerr=S, fmt='o', ms=4, alpha=0.8, label="KiDS Bin-A")
        plt.plot(R, oneh*A1h, lw=2, label=f"1-halo x {A1h:.2f}")
        plt.plot(R, twoh, lw=2, label="2-halo (PFGM)")
        plt.plot(R, mod, lw=2.2, label="total")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("R [Mpc]"); plt.ylabel(r"$\Delta\Sigma\ [M_\odot/{\rm kpc}^2]$")
        plt.title(f"Bin-A overlay ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/lensing/{out_tag}_overlay_{tag}.png", dpi=160)
        plt.close()

    print("FREE-A:", out_free)
    print("A1-LOCKED:", out_lock)


def main():
    ap = argparse.ArgumentParser(description="Covariance-aware Bin-A fit: free-A and A1-locked")
    ap.add_argument("--manifest", default="configs/run_manifest.yaml")
    args = ap.parse_args()
    man = load_manifest(args.manifest)
    fit_binA(man, out_tag="binA")

if __name__ == "__main__":
    main()
