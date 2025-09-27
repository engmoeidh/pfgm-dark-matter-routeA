import os, json, argparse, yaml
import numpy as np
import pandas as pd
from scipy import optimize, interpolate
import matplotlib.pyplot as plt

from src.pfgm.kernels import mu_lens, hankel_j2

# ----------------------------- helpers -----------------------------

def load_manifest(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_kids_clean(csv_path, cov_path=None):
    df = pd.read_csv(csv_path)
    # normalize to [R, DeltaSigma, DeltaSigma_err]
    cmap = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cmap:
                return cmap[n]
        raise KeyError(f"Missing columns in {csv_path}. Columns: {list(df.columns)}")
    Rcol  = pick("r")
    DScol = pick("deltasigma","delta_sigma","ds","delta_sigma_msun_kpc2")
    Ecol  = pick("deltasigma_err","delta_sigma_err","ds_err","sigma_deltasigma")
    df = df.rename(columns={Rcol:"R", DScol:"DeltaSigma", Ecol:"DeltaSigma_err"})
    R  = df["R"].values.astype(float)
    DS = df["DeltaSigma"].values.astype(float)
    S  = df["DeltaSigma_err"].values.astype(float)
    C  = None
    if cov_path and os.path.isfile(cov_path):
        try:
            C = np.loadtxt(cov_path, delimiter=",")
            if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != len(R):
                C = None
        except Exception:
            C = None
    return R, DS, S, C


def load_routeA_1h(csv_path, R_target):
    """Load Route-A 1-halo profile robustly and interpolate onto R_target.
       Accepts 2-col no-header (R, ΔΣ_1h), named columns, different delimiters.
       If only 'total' and '2h' exist, computes 1h = total - 2h."""
    import pandas as pd, numpy as np
    candidates_1h = [
        "deltasigma_1h","ds_1h","one_halo","1h","baryons_only","deltasigma1h","ds1h"
    ]
    candidates_R = ["r","radius_mpc","r_mpc"]
    def try_read(hdr):
        # sep=None lets pandas sniff delimiter (commas/semicolons/tabs)
        return pd.read_csv(csv_path, header=hdr, sep=None, engine="python")
    for hdr in [None, 0]:
        try:
            df = try_read(hdr)
        except Exception:
            continue
        # numeric only copy helps identify 2-col files
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # 2-col numeric → assume R, 1h
        if len(df.columns)==2 or len(numcols)==2:
            c0, c1 = (numcols if len(numcols)==2 else list(df.columns)[:2])
            Rtab = df[c0].astype(float).values
            oneh = df[c1].astype(float).values
            break
        cols = {c.lower().strip(): c for c in df.columns}
        # find R
        Rcol = None
        for k in candidates_R:
            if k in cols:
                Rcol = cols[k]; break
        if Rcol is None:
            # fallback: first column if numeric-looking
            Rcol = list(df.columns)[0]
        # find 1h
        Dcol = None
        for k in candidates_1h:
            if k in cols:
                Dcol = cols[k]; break
        if Dcol is None:
            # try to reconstruct from total and 2h
            tot = cols.get("total") or cols.get("deltasigma_total") or cols.get("ds_total")
            two = cols.get("2h") or cols.get("ds_2h") or cols.get("deltasigma_2h")
            if tot and two:
                oneh_series = df[tot].astype(float) - df[two].astype(float)
                Rtab = df[Rcol].astype(float).values
                oneh = oneh_series.values
                break
            else:
                # try last numeric column as 1h
                cand = None
                for c in df.columns[::-1]:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        cand = c; break
                if cand is not None:
                    Rtab = df[Rcol].astype(float).values
                    oneh = df[cand].astype(float).values
                    break
        else:
            Rtab = df[Rcol].astype(float).values
            oneh = df[Dcol].astype(float).values
            break
    else:
        raise ValueError(f"Cannot parse 1-halo CSV: {csv_path}")
    # Interpolate
    from scipy import interpolate
    f = interpolate.InterpolatedUnivariateSpline(Rtab, oneh, k=1, ext=1)
    return f(R_target)


def build_pnl(k_file, kmin, kmax, nk):
    df = pd.read_csv(k_file)
    cols = {c.lower(): c for c in df.columns}
    kcol = cols.get("k", list(df.columns)[0])
    pcol = cols.get("p_mpc3", cols.get("p", list(df.columns)[1]))
    ktab = df[kcol].values.astype(float)
    ptab = df[pcol].values.astype(float)
    mask = (ktab > 0) & (ptab > 0)
    kt, pt = ktab[mask], ptab[mask]
    ks = np.geomspace(max(kmin, kt.min()), min(kmax, kt.max()), nk)
    P  = np.interp(ks, kt, pt)
    return ks, P

def model_2h(R, k, Pnl, b0, b2, lam):
    Pgm = (b0 + b2*(k**2)) * mu_lens(k, lam) * Pnl
    return hankel_j2(k, Pgm, R)

def chi2(model, data, cov=None, sigma=None):
    r = model - data
    if cov is not None:
        try:
            inv = np.linalg.inv(cov)
            return float(r @ inv @ r)
        except Exception:
            pass
    if sigma is None:
        raise ValueError("Need sigma if covariance is not provided.")
    return float(np.sum((r/sigma)**2))

def aic_bic(k_params, chi2_val, n_points):
    """Compute AIC and BIC from chi2, number of params, and data points."""
    aic = chi2_val + 2*k_params
    bic = chi2_val + k_params*np.log(max(n_points,1))
    return float(aic), float(bic)

def choose_paths(man, bin_tag):
    bin_tag = bin_tag.upper()
    if bin_tag == "A":
        kids = man["paths"]["kids_binA_clean"]
        oneh = man["paths"]["routeA_1h_binA"]
    elif bin_tag == "B":
        kids = man["paths"]["kids_binB_clean"]
        oneh = man["paths"]["routeA_1h_binB"]
    else:
        raise ValueError("Unknown bin; use A or B")
    cov = man["paths"].get(f"kids_bin{bin_tag}_cov")
    return kids, oneh, cov

# ----------------------------- fitting -----------------------------

def fit_bin(man, out_tag="bin", bin_sel="A", do_gr_control=True):
    # Load inputs
    kids_path, oneh_path, cov_path = choose_paths(man, bin_sel)
    R, DS, S, C = load_kids_clean(kids_path, cov_path)
    oneh        = load_routeA_1h(oneh_path, R)
    k, Pnl      = build_pnl(man["paths"]["p_nl"],
                            man["fitting"]["k_min"], man["fitting"]["k_max"],
                            man["fitting"]["n_k"])

    # --- free-A fit: params [A1h, b0, b2, lam]
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
            try:
                L = np.linalg.cholesky(C)
                return np.linalg.solve(L, mod-DS)
            except Exception:
                pass
        return (mod-DS)/S

    res = optimize.least_squares(residual_free, x0, bounds=(lb, ub),
                                 xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    A1h_f, b0_f, b2_f, lam_f = res.x
    twoh_f = model_2h(R, k, Pnl, b0_f, b2_f, lam_f)
    mod_f  = A1h_f*oneh + twoh_f
    chi2_f = chi2(mod_f, DS, cov=C, sigma=S)
    nu_f   = len(R) - len(res.x)
    out_free = {
        "bin": bin_sel,
        "mode": "freeA",
        "A1h": float(A1h_f), "b0": float(b0_f), "b2": float(b2_f), "lam": float(lam_f),
        "chi2": float(chi2_f), "nu": int(nu_f), "chi2/nu": float(chi2_f/max(nu_f,1)),
    }

    # --- A1-locked (SPARC) fit: params [b0, b2, lam], A1h=1
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

    y0  = np.array([man["fitting"]["initial"]["b0"], man["fitting"]["initial"]["b2"], man["fitting"]["initial"]["lam"]])
    lb2 = np.array([man["fitting"]["bounds"]["b0"][0], man["fitting"]["bounds"]["b2"][0], man["fitting"]["bounds"]["lam"][0]])
    ub2 = np.array([man["fitting"]["bounds"]["b0"][1], man["fitting"]["bounds"]["b2"][1], man["fitting"]["bounds"]["lam"][1]])

    res2 = optimize.least_squares(residual_locked, y0, bounds=(lb2, ub2),
                                  xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    b0_l, b2_l, lam_l = res2.x
    twoh_l = model_2h(R, k, Pnl, b0_l, b2_l, lam_l)
    mod_l  = 1.0*oneh + twoh_l
    chi2_l = chi2(mod_l, DS, cov=C, sigma=S)
    nu_l   = len(R) - len(res2.x)
    out_lock = {
        "bin": bin_sel,
        "mode": "A1",
        "A1h": 1.0, "b0": float(b0_l), "b2": float(b2_l), "lam": float(lam_l),
        "chi2": float(chi2_l), "nu": int(nu_l), "chi2/nu": float(chi2_l/max(nu_l,1)),
    }

    # --- GR-kernel control (lam=0): params [A1h, b0, b2]
    def residual_gr(z):
        A1h, b0, b2 = z
        twoh_gr = model_2h(R, k, Pnl, b0, b2, 0.0)
        mod_gr  = A1h*oneh + twoh_gr
        if C is not None:
            try:
                L = np.linalg.cholesky(C)
                return np.linalg.solve(L, mod_gr-DS)
            except Exception:
                pass
        return (mod_gr-DS)/S

    z0  = [float(out_free['A1h']), float(out_free['b0']), float(out_free['b2'])]
    lbz = [man["fitting"]["bounds"]["A1h"][0], man["fitting"]["bounds"]["b0"][0], man["fitting"]["bounds"]["b2"][0]]
    ubz = [man["fitting"]["bounds"]["A1h"][1], man["fitting"]["bounds"]["b0"][1], man["fitting"]["bounds"]["b2"][1]]

    out_gr = None
    if do_gr_control:
        res_gr = optimize.least_squares(residual_gr, z0, bounds=(lbz, ubz),
                                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
        A1h_g, b0_g, b2_g = res_gr.x
        twoh_g = model_2h(R, k, Pnl, b0_g, b2_g, 0.0)
        mod_g  = A1h_g*oneh + twoh_g
        chi2_g = chi2(mod_g, DS, cov=C, sigma=S)
        nu_g   = len(R) - len(res_gr.x)
        out_gr = {
            "bin": bin_sel,
            "mode": "GRkernel",
            "A1h": float(A1h_g), "b0": float(b0_g), "b2": float(b2_g), "lam": 0.0,
            "chi2": float(chi2_g), "nu": int(nu_g), "chi2/nu": float(chi2_g/max(nu_g,1)),
        }

    # --- AIC/BIC and deltas
    aic_free, bic_free = aic_bic(4, out_free["chi2"], len(R))
    aic_lock, bic_lock = aic_bic(3, out_lock["chi2"], len(R))
    out_free["AIC"] = aic_free; out_free["BIC"] = bic_free
    out_lock["AIC"] = aic_lock; out_lock["BIC"] = bic_lock
    if out_gr is not None:
        aic_gr, bic_gr = aic_bic(3, out_gr["chi2"], len(R))
        out_gr["AIC"] = aic_gr; out_gr["BIC"] = bic_gr
        out_free["ΔAIC_vs_GR"] = aic_free - aic_gr; out_free["ΔBIC_vs_GR"] = bic_free - bic_gr
        out_lock["ΔAIC_vs_GR"] = aic_lock - aic_gr; out_lock["ΔBIC_vs_GR"] = bic_lock - bic_gr

    # --- Save outputs
    os.makedirs("results/tables", exist_ok=True)
    with open(f"results/tables/{out_tag}_{bin_sel}_fit_freeA.json","w") as f:
        json.dump(out_free, f, indent=2)
    with open(f"results/tables/{out_tag}_{bin_sel}_fit_A1.json","w") as f:
        json.dump(out_lock, f, indent=2)
    if out_gr is not None:
        with open(f"results/tables/{out_tag}_{bin_sel}_fit_GR.json","w") as f:
            json.dump(out_gr, f, indent=2)

    # --- Plots
    os.makedirs("figures/lensing", exist_ok=True)
    for tag, mod, twoh, A1h in [
        ("freeA", mod_f, twoh_f, A1h_f),
        ("A1",   mod_l, twoh_l, 1.0),
    ]:
        plt.figure(figsize=(6.2,4.6))
        plt.errorbar(R, DS, yerr=S, fmt='o', ms=4, alpha=0.8, label=f"KiDS Bin-{bin_sel}")
        plt.plot(R, oneh*A1h, lw=2, label=f"1-halo x {A1h:.2f}")
        plt.plot(R, twoh, lw=2, label="2-halo (PFGM)")
        plt.plot(R, mod, lw=2.2, label="total")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("R [Mpc]"); plt.ylabel(r"$\Delta\Sigma\ [M_\odot/{\rm kpc}^2]$")
        plt.title(f"Bin-{bin_sel} overlay ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/lensing/{out_tag}_{bin_sel}_overlay_{tag}.png", dpi=160)
        plt.close()
    if out_gr is not None:
        plt.figure(figsize=(6.2,4.6))
        plt.errorbar(R, DS, yerr=S, fmt='o', ms=4, alpha=0.8, label=f"KiDS Bin-{bin_sel}")
        plt.plot(R, oneh*out_gr["A1h"], lw=2, label=f"1-halo x {out_gr['A1h']:.2f}")
        twoh_g = model_2h(R, k, Pnl, out_gr["b0"], out_gr["b2"], 0.0)
        mod_g  = out_gr["A1h"]*oneh + twoh_g
        plt.plot(R, twoh_g, lw=2, label="2-halo (GR kernel)")
        plt.plot(R, mod_g, lw=2.2, label="total (GR)")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("R [Mpc]"); plt.ylabel(r"$\Delta\Sigma\ [M_\odot/{\rm kpc}^2]$")
        plt.title(f"Bin-{bin_sel} overlay (GR-kernel)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/lensing/{out_tag}_{bin_sel}_overlay_GR.png", dpi=160)
        plt.close()

    # --- print summaries
    print("FREE-A:", out_free)
    print("A1-LOCKED:", out_lock)
    if out_gr is not None:
        print("GR-KERNEL:", out_gr)

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Covariance-aware fit: free-A, A1-locked, and GR control")
    ap.add_argument("--manifest", default="configs/run_manifest.yaml")
    ap.add_argument("--bin", default="A")
    args = ap.parse_args()
    man = load_manifest(args.manifest)
    fit_bin(man, out_tag="bin", bin_sel=args.bin)

if __name__ == "__main__":
    main()
