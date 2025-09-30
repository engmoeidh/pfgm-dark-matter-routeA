import numpy as np, pandas as pd, argparse, os
import matplotlib.pyplot as plt

def hankel_j2(k, Pgm, R):
    # ΔΣ_2h(R) = ρ̄_m ∫ (k dk / 2π) P_gm(k) J2(kR)
    # build a log-integrator
    from scipy.special import jv
    out = []
    for r in R:
        J2 = jv(2, k*r)
        out.append(np.trapz(k*Pgm*J2/(2*np.pi), k))
    return np.array(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kids", default="data/raw/kids/KiDS_binA.csv")
    ap.add_argument("--pnl",  default="data/raw/Pnl_z0p286_HMcode.csv")
    ap.add_argument("--b0", type=float, default=1.6)
    ap.add_argument("--b2", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=1.2) # Mpc
    ap.add_argument("--z", type=float, default=0.286)
    args=ap.parse_args()

    dat = pd.read_csv(args.kids)   # columns: R [Mpc], DeltaSigma [Msun/kpc^2], err [...]
    R   = dat.iloc[:,0].to_numpy(float)
    D   = dat.iloc[:,1].to_numpy(float)
    E   = dat.iloc[:,2].to_numpy(float) if dat.shape[1]>2 else np.ones_like(D)

    tbl = pd.read_csv(args.pnl)    # columns: k [1/Mpc], P [Mpc^3]
    k = tbl.iloc[:,0].to_numpy(float)
    P = tbl.iloc[:,1].to_numpy(float)

    mu = 1.0/(1.0 + (k*args.lam)**2)          # PFGM lensing response
    Pgm = (args.b0 + args.b2*k**2) * mu * P   # Mpc^3

    # comoving rho_m
    h=0.674; Om=0.315
    rho_c0 = 2.775e11*(h**2)      # Msun/Mpc^3
    rho_m  = Om*rho_c0            # Msun/Mpc^3

    # Baseline (all comoving, no (1+z)^2)
    DS2 = rho_m * hankel_j2(k, Pgm, R)       # Msun/Mpc^2
    DS2 = DS2 / (1.0e6)                      # → Msun/kpc^2

    # Variants to test (multiplying factors)
    variants = {
      "baseline_comoving": DS2,
      "times_(1+z)^2": DS2 * (1+args.z)**2,
      "times_h2": DS2 * (h**2),
      "times_(1+z)^2*h2": DS2 * (1+args.z)**2 * (h**2)
    }

    for name, M in variants.items():
        resid = (M - D)/np.where(E>0,E,1.0)
        chi2  = float(np.sum(resid*resid))
        nu = len(R)-1
        print(f"{name:>18s}: chi2/nu ~ {chi2/nu:.2f}")

    # Optional: save a quick overlay
    os.makedirs("figures/diagnostics", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.errorbar(R, D, yerr=E, fmt='o', ms=4, alpha=0.8, label='data')
    for name,M in variants.items():
        plt.plot(R, M, label=name)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("R [Mpc] (comoving)"); plt.ylabel("ΔΣ [Msun/kpc²]")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("figures/diagnostics/kidsA_deltasigma2h_normcheck.png", dpi=150)
    print("Wrote figures/diagnostics/kidsA_deltasigma2h_normcheck.png")

if __name__=="__main__":
    main()
