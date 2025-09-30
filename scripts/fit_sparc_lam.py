import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from src.pfgm.routea_rc import load_struct_scales, routea_rc_model_sq

# --- Locations and outputs ---
CURVE_DIR = "data/raw/sparc/curves"
OUT_CSV   = "results/tables/sparc_lambda_summary.csv"
HIST_KPC  = "figures/sparc/sparc_lambda_hist.png"
HIST_MPC  = "figures/sparc/sparc_lambda_hist_mpc.png"
SUMMARY   = "results/tables/sparc_lambda_stats.csv"

# --- Master list loader (robust) ---
def load_sparc_master():
    """
    Try to read a SPARC master list to get galaxy names.
    If parsing fails or file missing, fall back to curve basenames in CURVE_DIR.
    """
    candidates = ["data/raw/sparc/SPARC_Table1_clean.csv",
                  "data/raw/sparc/SPARC_Lelli2016c.csv.txt"]
    for pth in candidates:
        if not os.path.isfile(pth):
            continue
        for kwargs in (dict(sep=None, engine="python"),
                       dict(sep=r"\s+", engine="python")):
            try:
                df = pd.read_csv(pth, **kwargs)
                df.columns = [str(c).strip() for c in df.columns]
                col = "Galaxy" if "Galaxy" in df.columns else ("galaxy" if "galaxy" in df.columns else None)
                if col:
                    return df[[col]].rename(columns={col:"Galaxy"}).drop_duplicates().reset_index(drop=True)
            except Exception:
                pass
    # fallback: whatever curves exist
    gals = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(CURVE_DIR,"*.csv"))]
    return pd.DataFrame({"Galaxy": gals})

# --- Per-galaxy curve loader ---
def load_curve(gal):
    direct = os.path.join(CURVE_DIR, f"{gal}.csv")
    if os.path.isfile(direct):
        return pd.read_csv(direct)
    for p in glob.glob(os.path.join(CURVE_DIR, "*.csv")):
        if os.path.splitext(os.path.basename(p))[0].lower() == gal.lower():
            return pd.read_csv(p)
    return None

# --- Multiplicative proxy used as fallback when no scales are available ---
def routeA_velocity_sq(R_kpc, Vbar_sq, eps, lam_kpc):
    """
    V_mod^2 = V_bar^2 * [1 + eps * (1 - exp(-R/lam))].
    eps >= 0, lam_kpc > 0.
    """
    R = np.asarray(R_kpc, dtype=float)
    return Vbar_sq * (1.0 + eps * (1.0 - np.exp(-R / lam_kpc)))

# --- Route-A RC fitter for one galaxy ---
def fit_one(gal, df, scales_cache):
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
    if Verr is not None:
        Verr = Verr[m]
    if len(R) < 6:
        raise ValueError("too few points")

    # Try to get components if present
    Vd = df.get("V_disk_kms", None)
    Vb = df.get("V_bulge_kms", None)
    Vg = df.get("V_gas_kms", None)
    Vd = Vd.values[m] if Vd is not None else None
    Vb = Vb.values[m] if Vb is not None else None
    Vg = Vg.values[m] if Vg is not None else None

    # If components aren't present, split Vbar heuristically (disc ~0.7, gas ~0.3)
    if Vd is None and Vg is None and Vb is None:
        Vd = 0.7 * Vba
        Vg = 0.3 * Vba
        Vb = np.zeros_like(Vba)

    # structural scales
    scales = scales_cache.get(gal, dict(Rd=None,Rb=None,Rg=None))

    Vbar2 = Vba**2
    Yobs  = Vob**2

    # --- velocity-space weighting ---
    if Verr is None:
        Verr = np.full_like(Vob, 10.0)  # km/s floor if missing
    sigmaV2 = np.clip(Verr, 5.0, None)**2
    wV = 1.0 / sigmaV2

    # Initial guesses and bounds
    eps0, lam0 = 0.2, 5.0
    lb = np.array([0.0, 0.3])
    ub = np.array([3.0, 200.0])

    def residual(p):
        eps, lam = p
        use_proxy = (scales.get("Rd") is None and scales.get("Rb") is None and scales.get("Rg") is None)
        if use_proxy:
            V2 = routeA_velocity_sq(R, Vbar2, eps, lam)
        else:
            V2 = routea_rc_model_sq(R, Vd, Vb, Vg, scales=scales, lam_kpc=lam, eps=eps)
        Vmod = np.sqrt(np.clip(V2, 0.0, None))
        return np.sqrt(wV) * (Vmod - Vob)

    res = optimize.least_squares(residual, x0=[eps0, lam0], bounds=(lb, ub),
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=8000)
    eps_fit, lam_fit = res.x
    r = residual(res.x)
    chi2 = float(np.sum(r*r))
    nu   = int(max(len(R)-len(res.x), 1))

    edge = "OK"
    lo, hi = lb[1], ub[1]
    if abs(lam_fit - hi) < 0.02*(hi-lo):
        edge = "EDGE_UP"
    if abs(lam_fit - lo) < 0.02*(hi-lo):
        edge = "EDGE_LOW"

    return dict(Galaxy=gal, lam_kpc=float(lam_fit), eps=float(eps_fit),
                chi2=chi2, nu=nu, status=edge)

# --- Main sweep ---
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(HIST_KPC), exist_ok=True)

    scales_cache = load_struct_scales()
    master = load_sparc_master()
    rows = []
    for gal in master["Galaxy"].tolist():
        df = load_curve(gal)
        if df is None:
            rows.append(dict(Galaxy=gal, lam_kpc=None, eps=None, chi2=None, nu=None, status="MISSING_CURVE")); continue
        try:
            rows.append(fit_one(gal, df, scales_cache))
        except Exception as e:
            rows.append(dict(Galaxy=gal, lam_kpc=None, eps=None, chi2=None, nu=None, status=f"ERROR: {e}"))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

    ok = out[(out["lam_kpc"].notna()) & (out["status"].str.startswith(("OK","EDGE")))]
    if len(ok) >= 2:
        # kpc
        plt.figure(figsize=(6.2,4.2))
        plt.hist(ok["lam_kpc"].values, bins=24, alpha=0.85, edgecolor="black")
        plt.xlabel("lambda_SPARC [kpc]"); plt.ylabel("Number of galaxies")
        plt.title("SPARC per-galaxy lambda (Route-A RC scaffold)")
        plt.tight_layout(); plt.savefig(HIST_KPC, dpi=160); plt.close()
        # Mpc
        lam_mpc = ok["lam_kpc"].values/1000.0
        plt.figure(figsize=(6.2,4.2))
        plt.hist(lam_mpc, bins=24, alpha=0.85, edgecolor="black")
        plt.xlabel("lambda_SPARC [Mpc]"); plt.ylabel("Number of galaxies")
        plt.title("SPARC per-galaxy lambda (Route-A RC scaffold)")
        plt.tight_layout(); plt.savefig(HIST_MPC, dpi=160); plt.close()
        print("Wrote", HIST_KPC, "and", HIST_MPC)
    else:
        print("Not enough OK/EDGE fits to plot histograms.")

    # stats
    edge_up = (out["status"]=="EDGE_UP").sum()
    okvals  = ok["lam_kpc"].to_numpy() if len(ok) else np.array([])
    stats = dict(N_total=int(len(out)), N_ok=int(len(ok)), N_edge_up=int(edge_up),
                 lam_kpc_median=float(np.median(okvals)) if okvals.size else None,
                 lam_kpc_p16=float(np.percentile(okvals,16)) if okvals.size else None,
                 lam_kpc_p84=float(np.percentile(okvals,84)) if okvals.size else None)
    pd.DataFrame([stats]).to_csv(SUMMARY, index=False)
    print("Wrote", SUMMARY)

if __name__ == "__main__":
    main()
