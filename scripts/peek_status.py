import os, json, glob, textwrap
from pathlib import Path
import numpy as np, pandas as pd

def sep(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def peek_csv(path, n=5):
    p = Path(path)
    if not p.is_file():
        print(f"[MISSING] {path}")
        return
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return
    print(f"[OK] {path}  shape={df.shape}")
    print("columns:", list(df.columns))
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df.head(n))
    # quick NaN summary for common fields
    for key in ("lambda_mpc","lam_kpc","chi2_red","thetaE_GR","thetaE_PFGM"):
        if key in df.columns:
            nonnan = df[key].notna().sum()
            print(f"  non-NaN {key}: {nonnan}/{len(df)}")
    return df

def peek_jsons(pattern, limit=3):
    hits = sorted(glob.glob(pattern))
    if not hits:
        print(f"[MISSING] {pattern}")
        return
    for i,p in enumerate(hits[:limit]):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            print(f"[OK] {p} keys:", sorted(d.keys()))
            for k in ("lambda_mpc","best_lambda","lambda_best","lambda","lam","chi2_red","chi2/nu","AIC","BIC"):
                if k in d: print(f"   {k} = {d[k]}")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

def peek_text(path, n=2000):
    p = Path(path)
    if not p.is_file():
        print(f"[MISSING] {path}")
        return
    s = p.read_text(encoding="utf-8", errors="ignore")
    print(f"[OK] {path}  len={len(s)}")
    print(s[:n])

def peek_h5(path):
    try:
        import h5py
    except Exception:
        print("[NOTE] h5py not installed; skipping HDF5 peek.")
        return
    p = Path(path)
    if not p.is_file():
        print(f"[MISSING] {path}")
        return
    with h5py.File(p,"r") as h:
        print(f"[OK] {path}")
        try:
            z = h["z"][:]; k = h["k"][:]
            print(f"  datasets: z({z.shape}) k({k.shape}) P({h['P'].shape})")
            print(f"  z[0]={z[0]:.3f}, nk={len(k)}  kmin={k.min():.3g} kmax={k.max():.3g}")
        except Exception as e:
            print("  could not read standard datasets:", e)

def main():
    sep("SPARC: per-galaxy master & fit summary")
    peek_csv("results/tables/sparc_lambda_summary.csv")
    dfm = peek_csv("results/tables/per_galaxy_master.csv")
    if dfm is not None and {"lam_kpc","chi2_red"}.issubset(dfm.columns):
        ok = dfm.query("flag_ok==1 and lam_kpc.notna()")
        if len(ok):
            print(f"  chi2/nu median (OK rows): {np.nanmedian(ok['chi2_red']):.3f}")
            q16,q84 = np.percentile(ok["lam_kpc"],[16,84])
            print(f"  lambda[kpc] median/p16/p84: {np.median(ok['lam_kpc']):.3g}, {q16:.3g}, {q84:.3g}")

    sep("Lensing: KiDS/SDSS summaries & joint λ table")
    peek_csv("results/tables/KiDS_fit_summary_FINAL.csv")
    peek_csv("results/tables/SDSS_fit_summary_FINAL.csv")
    peek_csv("results/tables/lam_table_all.csv")
    sep("SDSS/KiDS JSON samples")
    peek_jsons("results/tables/bin_*LOWZ*json", limit=2)
    peek_jsons("results/tables/bin_*CMASS*json", limit=2)

    sep("SLACS: compare CSV & final table")
    peek_csv("data/raw/slacs_compare_autogen.csv")
    peek_csv("results/tables/SLACS_PFGM_FINAL.csv")

    sep("Cosmology: Pm grid & 3x2pt summary")
    peek_h5("results/pm/PFGM_Pm_grid.hdf5")
    peek_text("results/3x2pt/3x2pt_summary.txt", n=500)

    sep("Guards (if present)")
    peek_csv("data/raw/guards_summary.csv")
    peek_text("results/guards/APPENDIX_GUARDS.csv")

    sep("Posterior (quick integrity)")
    try:
        d = np.load("results/posteriors/alpha_joint_posterior.npz")
        lam, post = d["lambda_mpc"], d["posterior_lambda"]
        print(f"[OK] posterior arrays: lam({lam.shape}), post({post.shape})")
        try:
            norm = float(np.trapz(post, lam))
        except Exception:
            norm = float("nan")
        print(f"  int_p_dlambda = {norm}")
        if np.all(np.isfinite(post)) and np.all(post>=0) and np.isfinite(norm) and norm>0:
            print(f"  lambda_mode ~= {float(lam[np.argmax(post)])}")
        else:
            print("  posterior has NaN/negatives or zero/NaN norm (will fix next).")
    except Exception as e:
        print("[MISSING/ERROR] results/posteriors/alpha_joint_posterior.npz:", e)

if __name__ == "__main__":
    main()
