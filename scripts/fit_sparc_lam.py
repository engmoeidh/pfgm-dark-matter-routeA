import os, glob, json
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

CURVE_DIR = "data/raw/sparc/curves"
OUT_CSV   = "results/tables/sparc_lambda_summary.csv"
HIST_KPC  = "figures/sparc/sparc_lambda_hist.png"
HIST_MPC  = "figures/sparc/sparc_lambda_hist_mpc.png"
SUMMARY   = "results/tables/sparc_lambda_stats.csv"

def load_sparc_master():
    # Try a master list; otherwise fall back to whatever curves exist
    candidates = ["data/raw/sparc/SPARC_Table1_clean.csv",
                  "data/raw/sparc/SPARC_Lelli2016c.csv.txt"]
    for p in candidates:
        if os.path.isfile(p):
            for kwargs in (dict(sep=None, engine="python"),
                           dict(sep=r"\s+", engine="python")):
                try:
                    df = pd.read_csv(p, **kwargs)
                    df.columns = [str(c).strip() for c in df.columns]
                    col = "Galaxy" if "Galaxy" in df.columns else ("galaxy" if "galaxy" in df.columns else None)
                    if col:
                        return df[[col]].rename(columns={col:"Galaxy"}).drop_duplicates().reset_index(drop=True)
                except Exception:
                    pass
    gals = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(CURVE_DIR,"*.csv"))]
    return pd.DataFrame({"Galaxy": gals})

def load_curve(gal):
    direct = os.path.join(CURVE_DIR, f"{gal}.csv")
    if os.path.isfile(direct):
        return pd.read_csv(direct)
    for p in glob.glob(os.path.join(CURVE_DIR, "*.csv")):
        if os.path.splitext(os.path.basename(p))[0].lower() == gal.lower():
            return pd.read_csv(p)
    return None


def routeA_velocity_sq(R_kpc, Vbar_sq, eps, lam_kpc):
    """
    Route-A rotational proxy: V_mod^2 = V_bar^2 * [1 + eps * (1 - exp(-R/lam))]
    Inputs:
      R_kpc   : radii in kpc  (array)
      Vbar_sq : baryonic velocity squared (km/s)^2 (array)
      eps     : dimensionless strength >= 0
      lam_kpc : range in kpc
    """
    import numpy as np
    R = np.asarray(R_kpc, dtype=float)
    return Vbar_sq * (1.0 + eps * (1.0 - np.exp(-R / lam_kpc)))

def fit_one(gal, df):
    import numpy as np
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["r_kpc","v_obs_kms","v_baryon_kms"]
    for n in need:
        if n not in cols:
            raise ValueError("missing column: " + n)

    R   = df[cols["r_kpc"]].astype(float).values
    Vob = df[cols["v_obs_kms"]].astype(float).values
    Vba = df[cols["v_baryon_kms"]].astype(float).values
    Verr = df[cols["v_err_kms"]].astype(float).values if "v_err_kms" in cols else None

    m = np.isfinite(R) & np.isfinite(Vob) & np.isfinite(Vba) & (R>0)
    R, Vob, Vba = R[m], Vob[m], Vba[m]
    if Verr is not None: Verr = Verr[m]
    if len(R) < 6:
        raise ValueError("too few points")

    Vbar2 = Vba**2
    Yobs  = Vob**2

    if Verr is not None:
        # var(V^2) ~ (2 V σ_V)^2  (propagate obs errors)
        sigma2 = (2.0 * Vob * Verr)**2
        w = 1.0 / np.clip(sigma2, 1e-12, None)
    else:
        w = np.ones_like(Yobs)

    # Initial guesses: small eps, mid-range lambda
    eps0, lam0 = 0.2, 5.0  # lam in kpc
    # Bounds: eps ∈ [0, 3], lam ∈ [0.3, 200]  (broad; edge will be flagged)
    lb = np.array([0.0, 0.3])
    ub = np.array([3.0, 200.0])

    def residual(p):
        eps, lam = p
        V2 = routeA_velocity_sq(R, Vbar2, eps, lam)
        return np.sqrt(w) * (V2 - Yobs)

    from scipy import optimize
    res = optimize.least_squares(residual, x0=[eps0, lam0], bounds=(lb, ub),
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=8000)
    eps_fit, lam_fit = res.x
    chi2 = float(np.sum((residual(res.x))**2))
    nu   = int(max(len(R)-len(res.x), 1))

    edge = "OK"
    if abs(lam_fit - ub[1]) < 0.02*(ub[1]-lb[1]): edge = "EDGE_UP"
    if abs(lam_fit - lb[1]) < 0.02*(ub[1]-lb[1]): edge = "EDGE_LOW"

    return dict(Galaxy=gal, lam_kpc=float(lam_fit), eps=float(eps_fit), chi2=chi2, nu=nu, status=edge)
(gal, df):
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["r_kpc","v_obs_kms","v_baryon_kms"]
    if any(n not in cols for n in need):
        raise ValueError("missing columns: " + ",".join(need))
    R   = df[cols["r_kpc"]].astype(float).values
    Vob = df[cols["v_obs_kms"]].astype(float).values
    Vba = df[cols["v_baryon_kms"]].astype(float).values
    Verr = df[cols["v_err_kms"]].astype(float).values if "v_err_kms" in cols else None

    m = np.isfinite(R) & np.isfinite(Vob) & np.isfinite(Vba) & (R>0)
    R, Vob, Vba = R[m], Vob[m], Vba[m]
    if Verr is not None: Verr = Verr[m]
    if len(R) < 6:
        raise ValueError("too few points")

    Y = Vob**2 - Vba**2
    if Verr is not None:
        sigma2 = (2*Vob*Verr)**2
        w = 1.0/np.clip(sigma2, 1e-12, None)
    else:
        w = np.ones_like(Y)

    A0   = max(np.median(np.clip(Y,0,None)), 1.0) * max(np.median(R), 1.0)
    lam0 = 5.0
    bounds = ([0.0, 0.3], [np.inf, 200.0])  # allow up to 200 kpc; we flag edges

    def residual(p):
        A, lam = p
        V2 = yukawa_velocity_sq(R, A, lam)
        return np.sqrt(w) * (V2 - Y)

    res = optimize.least_squares(residual, x0=[A0, lam0], bounds=bounds,
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=8000)
    Afit, lamfit = res.x
    chi2 = float(np.sum((residual(res.x))**2))
    nu   = int(max(len(R)-len(res.x), 1))
    edge = "OK"
    lo, hi = bounds[0][1], bounds[1][1]
    if abs(lamfit-hi) < 0.02*(hi-lo):
        edge = "EDGE_UP"
    if abs(lamfit-lo) < 0.02*(hi-lo):
        edge = "EDGE_LOW"
    return dict(Galaxy=gal, lam_kpc=float(lamfit), A=float(Afit), chi2=chi2, nu=nu, status=edge)

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(HIST_KPC), exist_ok=True)

    master = load_sparc_master()
    rows = []
    for gal in master["Galaxy"].tolist():
        df = load_curve(gal)
        if df is None:
            rows.append(dict(Galaxy=gal, lam_kpc=None, A=None, chi2=None, nu=None, status="MISSING_CURVE")); continue
        try:
            rows.append(fit_one(gal, df))
        except Exception as e:
            rows.append(dict(Galaxy=gal, lam_kpc=None, A=None, chi2=None, nu=None, status=f"ERROR: {e}"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

    ok = out[(out["lam_kpc"].notna()) & (out["status"].str.startswith("OK"))]
    if len(ok) >= 2:
        plt.figure(figsize=(6.2,4.2))
        plt.hist(ok["lam_kpc"].values, bins=24, alpha=0.85, edgecolor="k")
        plt.xlabel(r"$\lambda_{\rm SPARC}$ [kpc]"); plt.ylabel("Number of galaxies")
        plt.title("SPARC per-galaxy λ (phenomenological Yukawa fit)")
        plt.tight_layout(); plt.savefig(HIST_KPC, dpi=160); plt.close()
        # Mpc histogram
        lam_mpc = ok["lam_kpc"].values/1000.0
        plt.figure(figsize=(6.2,4.2))
        plt.hist(lam_mpc, bins=24, alpha=0.85, edgecolor="k")
        plt.xlabel(r"$\lambda_{\rm SPARC}$ [Mpc]"); plt.ylabel("Number of galaxies")
        plt.title("SPARC per-galaxy λ (Mpc)")
        plt.tight_layout(); plt.savefig(HIST_MPC, dpi=160); plt.close()
        print("Wrote", HIST_KPC, "and", HIST_MPC)
    else:
        print("Not enough OK fits to plot histograms.")

    # compact stats
    edge_up = (out["status"]=="EDGE_UP").sum()
    okvals = ok["lam_kpc"].to_numpy() if len(ok) else np.array([])
    stats = dict(N_total=int(len(out)), N_ok=int(len(ok)), N_edge_up=int(edge_up),
                 lam_kpc_median=float(np.median(okvals)) if okvals.size else None,
                 lam_kpc_p16=float(np.percentile(okvals,16)) if okvals.size else None,
                 lam_kpc_p84=float(np.percentile(okvals,84)) if okvals.size else None)
    pd.DataFrame([stats]).to_csv(SUMMARY, index=False)
    print("Wrote", SUMMARY)

if __name__ == "__main__":
    main()
