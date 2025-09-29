import argparse, os
import numpy as np
import pandas as pd

def kde_gauss(x, xs, sig):
    sig = np.maximum(np.asarray(sig, float), 1e-12)
    xs  = np.asarray(xs, float)
    x   = np.asarray(x, float)
    dx  = (x[:,None] - xs[None,:]) / sig[None,:]
    ker = np.exp(-0.5*dx*dx) / (np.sqrt(2*np.pi)*sig)
    return ker.sum(axis=1)

def load_lambda_from_sparc(path):
    df = pd.read_csv(path)
    ok = df["lam_kpc"].notna()
    lam = df.loc[ok, "lam_kpc"].to_numpy(float) / 1000.0  # kpc -> Mpc
    if lam.size == 0:
        return np.array([]), np.array([])
    # default width: 0.3*lambda (log-ish)
    sig = 0.3*np.maximum(lam, 1e-6)
    return lam, sig

def load_lambda_from_lensing(path):
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    df = pd.read_csv(path)
    if "lambda_mpc" not in df.columns:
        return np.array([]), np.array([])
    lam = df["lambda_mpc"].to_numpy(float)
    em  = df["lambda_err_minus"].to_numpy(float) if "lambda_err_minus" in df.columns else np.full_like(lam, np.nan)
    ep  = df["lambda_err_plus"].to_numpy(float)  if "lambda_err_plus"  in df.columns else np.full_like(lam, np.nan)
    # use RMS of asym errors; fallback to 0.3*lambda
    sig = 0.5*(np.abs(em)+np.abs(ep))
    sig = np.where(np.isfinite(sig) & (sig>0), sig, 0.3*np.maximum(lam, 1e-6))
    return lam, sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparc", required=True)
    ap.add_argument("--lensing", required=True)
    ap.add_argument("--slacs", required=False)   # unused in scaffold
    ap.add_argument("--cosmo", required=False)   # unused in scaffold
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    np.random.seed(args.seed)

    lam_s, sig_s = load_lambda_from_sparc(args.sparc)
    lam_l, sig_l = load_lambda_from_lensing(args.lensing)

    lam_all = np.concatenate([lam_s, lam_l])
    sig_all = np.concatenate([sig_s, sig_l]) if lam_l.size else sig_s

    # Filter to positive, finite values only
    m = (lam_all > 0) & np.isfinite(lam_all) & np.isfinite(sig_all)
    lam_all, sig_all = lam_all[m], sig_all[m]
    if lam_all.size == 0:
        raise SystemExit("No usable lambda>0 for posterior after filtering.")

    # Robust log grid
    xmin = max(1e-4, float(lam_all.min())/5.0)
    xmax = max(xmin*10.0, float(lam_all.max())*5.0)
    x    = np.logspace(np.log10(xmin), np.log10(xmax), 1024)

    # KDE and safe normalization
    post = kde_gauss(x, lam_all, sig_all)
    post = np.maximum(post, 0.0)
    norm = float(np.trapz(post, x))
    if not (np.isfinite(norm) and norm > 0):
        # widen kernels and retry once
        sig_all = np.maximum(sig_all, 0.5*np.maximum(lam_all, 1e-6))
        post = kde_gauss(x, lam_all, sig_all)
        post = np.maximum(post, 0.0)
        norm = float(np.trapz(post, x))
    post = post / (norm if norm > 0 else 1.0)

    # For completeness, export an alpha proxy ~ lambda^4 (unit-agnostic here)
    alpha = x**4
    # Change-of-variables proxy; keep unit-agnostic normalization
    dadt = np.gradient(alpha, x)
    post_alpha = post / np.maximum(dadt, 1e-30)
    post_alpha = np.maximum(post_alpha, 0.0)
    na = float(np.trapz(post_alpha, alpha))
    post_alpha = post_alpha / (na if na > 0 else 1.0)

    # Save
    np.savez(args.out,
             lambda_mpc=x,
             posterior_lambda=post,
             alpha_km4=alpha,
             posterior_alpha=post_alpha)

    # Figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.0,4.0))
    plt.loglog(x, post, lw=2)
    plt.xlabel("lambda [Mpc]")
    plt.ylabel("posterior (norm.)")
    plt.tight_layout(); os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    plt.savefig(args.fig, dpi=160); plt.close()

    print("Wrote", args.out, "and", args.fig)

if __name__ == "__main__":
    main()
