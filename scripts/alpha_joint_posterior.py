import argparse, os, json
import numpy as np, pandas as pd

def kde_gauss(x, xs, sig):
    sig = np.maximum(sig, 1e-9)
    dx = (x[:,None] - xs[None,:]) / sig[None,:]
    return np.sum(np.exp(-0.5*dx*dx) / (np.sqrt(2*np.pi)*sig), axis=1)

def load_lambda_from_sparc(p):
    df = pd.read_csv(p)
    ok = df["lam_kpc"].notna() & df.get("flag_ok",0).astype(int).eq(1)
    lam = df.loc[ok,"lam_kpc"].to_numpy(dtype=float)/1000.0  # → Mpc
    # naive width: 0.3 dex fraction for placeholder
    sig = 0.3*np.maximum(lam, 1e-6)
    return lam, sig

def load_lambda_from_lensing(p):
    if not os.path.isfile(p): return np.array([]), np.array([])
    df = pd.read_csv(p)
    lam = df["lambda_mpc"].to_numpy(dtype=float)
    # asymmetric errors → RMS as proxy
    em = np.nan_to_num(df.get("lambda_err_minus", np.full_like(lam,np.nan)).to_numpy(dtype=float), nan=np.nanmedian(lam)*0.2)
    ep = np.nan_to_num(df.get("lambda_err_plus",  np.full_like(lam,np.nan)).to_numpy(dtype=float), nan=np.nanmedian(lam)*0.2)
    sig = 0.5*(np.abs(em)+np.abs(ep))
    sig = np.where(np.isfinite(sig) & (sig>0), sig, 0.3*np.maximum(lam,1e-6))
    return lam, sig

def maybe_dummy_array(p):
    return np.array([]), np.array([])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparc", required=True)
    ap.add_argument("--lensing", required=True)
    ap.add_argument("--slacs", required=False)
    ap.add_argument("--cosmo", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--nchains", type=int, default=4)
    ap.add_argument("--ndraws", type=int, default=4000)
    ap.add_argument("--treat-edge-as-lower", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Collect λ measurements (Mpc)
    lam_s, sig_s = load_lambda_from_sparc(args.sparc)
    lam_l, sig_l = load_lambda_from_lensing(args.lensing)
    # For this scaffold, we ignore SLACS/cosmo or treat them as empty
    lam_all = np.concatenate([lam_s, lam_l])
    sig_all = np.concatenate([sig_s, sig_l])
    if lam_all.size == 0:
        raise SystemExit("No λ constraints found in SPARC/lensing; cannot build posterior.")

    # Build a simple 1D posterior over λ (Mpc) on a log grid
    x = np.logspace(np.log10(max(1e-3, np.min(lam_all)/5.0)),
                    np.log10(np.max(lam_all)*5.0), 1024)
    post = kde_gauss(x, lam_all, sig_all)
    post = np.maximum(post, 1e-300)
    post /= np.trapz(post, x)

    # Placeholder α ∝ λ^4 (units suppressed). We export both λ and α views.
    alpha = x**4
    post_alpha = post / np.trapz(post * np.gradient(alpha, x), alpha)  # change-of-variables (approx.)

    # Save
    np.savez(args.out, lambda_mpc=x, posterior_lambda=post, alpha_km4=alpha, posterior_alpha=post_alpha)

    # Figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4,4.2))
    plt.loglog(x, post, lw=2)
    plt.xlabel(r"$\lambda\ {\rm [Mpc]}$")
    plt.ylabel("Posterior (arb. norm.)")
    plt.title("Unified λ posterior (scaffold)")
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.fig, dpi=160); plt.close()

    print(f"Wrote {args.out} and {args.fig} (scaffold posterior)")
if __name__ == "__main__":
    main()
