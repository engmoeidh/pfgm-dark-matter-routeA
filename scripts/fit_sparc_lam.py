import os, glob, json
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


CURVE_DIR = "data/raw/sparc/curves"
OUT_CSV   = "results/tables/sparc_lambda_summary.csv"
HIST_PNG  = "figures/sparc/sparc_lambda_hist.png"

def load_sparc_master():
    """
    Try to read a SPARC master list to get galaxy names.
    If that fails (parser errors / missing column), fall back to the curves directory.
    """
    import pandas as pd, glob, os
    candidates = [
        "data/raw/sparc/SPARC_Table1_clean.csv",
        "data/raw/sparc/SPARC_Lelli2016c.csv.txt",
    ]
    for pth in candidates:
        if not os.path.isfile(pth):
            continue
        # try flexible CSV
        for kwargs in (dict(sep=None, engine="python"), dict(delim_whitespace=True)):
            try:
                df = pd.read_csv(pth, **kwargs)
                # normalize header strings
                df.columns = [str(c).strip() for c in df.columns]
                if "Galaxy" in df.columns:
                    return df[["Galaxy"]].drop_duplicates().reset_index(drop=True)
                # SPARC Lelli tables sometimes have a "galaxy" lowercase
                if "galaxy" in df.columns:
                    df = df.rename(columns={"galaxy":"Galaxy"})
                    return df[["Galaxy"]].drop_duplicates().reset_index(drop=True)
            except Exception:
                continue
    # Fallback: build list from curve files present
    gals = []
    for pth in glob.glob(os.path.join(CURVE_DIR, "*.csv")):
        base = os.path.splitext(os.path.basename(pth))[0]
        gals.append(base)
    if gals:
        import pandas as pd
        return pd.DataFrame({"Galaxy": gals})
    raise FileNotFoundError("No SPARC master table and no per-galaxy curves found.")

def load_curve(gal):
    """
    Try exact match, else case-insensitive search in CURVE_DIR.
    """
    import pandas as pd, os, glob
    direct = os.path.join(CURVE_DIR, f"{gal}.csv")
    if os.path.isfile(direct):
        return pd.read_csv(direct)
    # case-insensitive search
    gal_l = gal.lower()
    for pth in glob.glob(os.path.join(CURVE_DIR, "*.csv")):
        stem = os.path.splitext(os.path.basename(pth))[0]
        if stem.lower() == gal_l:
            return pd.read_csv(pth)
    return None


def load_curve(gal):
    # Expect curves as data/raw/sparc/curves/<Galaxy>.csv (case-insensitive fallback)
    direct = os.path.join(CURVE_DIR, f"{gal}.csv")
    if os.path.isfile(direct):
        df = pd.read_csv(direct)
        return df
    # case-insensitive search
    for p in glob.glob(os.path.join(CURVE_DIR, "*.csv")):
        if os.path.splitext(os.path.basename(p))[0].lower() == gal.lower():
            return pd.read_csv(p)
    return None

def yukawa_velocity_sq(R_kpc, A, lam_kpc):
    # Simple phenomenological add-on: V_add^2 ~ A * exp(-R/lam) / R
    # Dimensions: A in (km/s)^2 * kpc, R in kpc, lam in kpc.
    R = np.asarray(R_kpc, dtype=float)
    return A * np.exp(-R/lam_kpc) / np.clip(R, 1e-6, None)

def fit_one(gal, df):
    # Expect Galaxy,R_kpc,V_obs_kms,V_baryon_kms
    cols = {c.lower().strip(): c for c in df.columns}
    for needed in ["r_kpc","v_obs_kms","v_baryon_kms"]:
        if needed not in cols:
            raise ValueError(f"{gal}: missing column '{needed}' in {list(df.columns)}")
    R = df[cols["r_kpc"]].astype(float).values
    Vobs = df[cols["v_obs_kms"]].astype(float).values
    Vbar = df[cols["v_baryon_kms"]].astype(float).values

    # Clean NaNs/infs
    m = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
    R, Vobs, Vbar = R[m], Vobs[m], Vbar[m]
    if len(R) < 6:
        raise ValueError(f"{gal}: too few points after cleaning")

    # Fit A, lam to minimize residual of V^2
    Y = Vobs**2 - Vbar**2
    # Initial guesses
    A0  = np.median(np.clip(Y, 0, None)) * max(np.median(R), 1.0)
    lam0 = 5.0     # kpc-scale starting value; we bound to 0.5–50 kpc by default

    def residual(p):
        A, lam = p
        if lam <= 0.1:
            return 1e9 * np.ones_like(Y)
        V2 = yukawa_velocity_sq(R, A, lam)
        return (V2 - Y)

    bounds = ([0.0, 0.5], [np.inf, 50.0])  # A>=0; 0.5 <= lam <= 50 kpc
    res = optimize.least_squares(residual, x0=[A0, lam0], bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
    Afit, lamfit = res.x
    chi2 = np.sum(res.fun**2)
    dof  = len(R) - len(res.x)
    return float(lamfit), float(Afit), float(chi2), int(max(dof,0))

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(HIST_PNG), exist_ok=True)

    master = load_sparc_master()
    rows = []
    for gal in master["Galaxy"].tolist():
        df = load_curve(gal)
        if df is None:
            rows.append({"Galaxy": gal, "lam_kpc": None, "A": None, "chi2": None, "nu": None, "status": "MISSING_CURVE"})
            continue
        try:
            lam, A, chi2, nu = fit_one(gal, df)
            rows.append({"Galaxy": gal, "lam_kpc": lam, "A": A, "chi2": chi2, "nu": nu, "status": "OK"})
        except Exception as e:
            rows.append({"Galaxy": gal, "lam_kpc": None, "A": None, "chi2": None, "nu": None, "status": f"ERROR: {e}"})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

    # Histogram of fitted λ for OK entries
    ok = out[(out["status"]=="OK") & out["lam_kpc"].notna()]
    if len(ok) >= 2:
        plt.figure(figsize=(6.2,4.2))
        plt.hist(ok["lam_kpc"].values, bins=20, alpha=0.85, edgecolor="k")
        plt.xlabel(r"$\lambda_{\rm SPARC}$ [kpc]")
        plt.ylabel("Number of galaxies")
        plt.title("SPARC per-galaxy λ (phenomenological Yukawa fit)")
        plt.tight_layout()
        plt.savefig(HIST_PNG, dpi=160)
        plt.close()
        print("Wrote", HIST_PNG)
    else:
        print("Not enough OK fits to plot histogram.")

if __name__ == "__main__":
    main()
