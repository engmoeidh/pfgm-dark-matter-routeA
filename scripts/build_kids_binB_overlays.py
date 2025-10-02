import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fitjson', required=True)
parser.add_argument('--label', default='')
args = parser.parse_args()
import numpy as np, pandas as pd, os, json
from pathlib import Path
from scipy.optimize import least_squares
from scipy.special import jv
import matplotlib.pyplot as plt

# --- constants / conventions ---
H = 0.674
OM = 0.315
RHO_C0 = 2.775e11 * H**2      # Msun/Mpc^3
RHO_M  = OM * RHO_C0

# exact J2 Hankel
def hankel_j2(k, Pgm, R):
    out = []
    for r in R:
        out.append(np.trapz(k * Pgm * jv(2, k*r) / (2*np.pi), k))
    return np.array(out)

def load_and_align_1h(kids_csv, route_csv):
    d  = pd.read_csv(kids_csv).to_numpy(float)
    Rk = d[:,0]
    oneh = pd.read_csv(route_csv).to_numpy(float)
    R1, D1 = oneh[:,0], oneh[:,1]
    # interpolate in log R for robustness
    idx = np.argsort(R1); R1, D1 = R1[idx], D1[idx]
    DS1 = np.interp(np.log(np.clip(Rk,1e-6,None)),
                    np.log(np.clip(R1,1e-6,None)), D1)
    return Rk, DS1

def model_2h(R, k, P, b0, b2, lam, z):
    mu  = 1.0/(1.0 + (k*lam)**2) if lam>0 else np.ones_like(k)
    Pgm = (b0 + b2*k*k) * mu * P
    DS2 = RHO_M * hankel_j2(k, Pgm, R) / 1e6        # Msun/kpc^2
    DS2 *= (1.0 + z)**2                              # KiDS convention
    return DS2

def fit_case(kids_csv, route_csv, pnl_csv, z, case):
    # case: "FREEA" (fit A1h,b0,b2,lam), "A1" (A1h=1; fit b0,b2,lam), "GR" (lam=0; fit A1h,b0,b2)
    data = pd.read_csv(kids_csv).to_numpy(float)
    R, D, E = data[:,0], data[:,1], data[:,2]
    kP = pd.read_csv(pnl_csv).to_numpy(float)
    k, P = kP[:,0], kP[:,1]
    R1h, DS1 = load_and_align_1h(kids_csv, route_csv)

    assert np.allclose(R, R1h), "R grids differ after alignment — unexpected"

    if case=="FREEA":
        p0  = np.array([ 10.0,  1.6, 10.0, 1.0])   # A1h, b0, b2, lam
        lo  = np.array([  0.0,  0.3,  0.0, 0.05])
        hi  = np.array([1e5 ,   5.0, 50.0, 10.0])
        def resid(p):
            A1h, b0, b2, lam = p
            M = A1h*DS1 + model_2h(R, k, P, b0, b2, lam, z)
            return (M - D)/np.where(E>0,E,1.0)
        kpar = 4

    elif case=="A1":
        p0  = np.array([ 1.6, 10.0, 1.0])         # b0, b2, lam
        lo  = np.array([ 0.3,  0.0, 0.05])
        hi  = np.array([ 5.0, 50.0, 10.0])
        def resid(p):
            b0, b2, lam = p
            M = 1.0*DS1 + model_2h(R, k, P, b0, b2, lam, z)
            return (M - D)/np.where(E>0,E,1.0)
        kpar = 3

    elif case=="GR":
        p0  = np.array([ 10.0, 1.6, 10.0])        # A1h, b0, b2  (lam fixed = 0)
        lo  = np.array([  0.0, 0.3,  0.0])
        hi  = np.array([1e5 ,  5.0, 50.0])
        def resid(p):
            A1h, b0, b2 = p
            M = A1h*DS1 + model_2h(R, k, P, b0, b2, 0.0, z)
            return (M - D)/np.where(E>0,E,1.0)
        kpar = 3
    else:
        raise ValueError("unknown case")

    fit = least_squares(resid, p0, bounds=(lo,hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=30000)
    r   = resid(fit.x); chi2 = float(np.sum(r*r)); nu = max(len(R)-kpar, 1)
    if case=="FREEA":
        A1h, b0, b2, lam = fit.x
    elif case=="A1":
        b0, b2, lam = fit.x; A1h = 1.0
    else:
        A1h, b0, b2 = fit.x; lam = 0.0

    # build curves
    DS2 = model_2h(R, k, P, b0, b2, lam, z)
    M   = A1h*DS1 + DS2
    pars = dict(A1h=float(A1h), b0=float(b0), b2=float(b2), lambda_mpc=float(lam),
                chi2_red=float(chi2/nu), chi2=float(chi2), nu=int(nu))
    return R, D, E, DS1, DS2, M, pars

def save_overlay(R, D, E, DS1, DS2, M, title, outpath):
    plt.figure(figsize=(6.6,4.6))
    plt.errorbar(R, D, yerr=E, fmt='o', ms=4, capsize=2, label="data")
    plt.plot(R, DS1, lw=1.0, label="1h (unscaled)")
    # scaled 1h: infer A1h by least-squares on (M-DS2) vs DS1 at the given R
    num = np.sum((M-DS2)*DS1); den = np.sum(DS1*DS1); Aeff = num/den if den>0 else 0.0
    plt.plot(R, Aeff*DS1, lw=1.5, label=f"1h scaled (A≈{Aeff:.2g})")
    plt.plot(R, DS2, lw=1.5, label="2h")
    plt.plot(R, M,  lw=2.0, label="total")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("R [Mpc] (comoving)"); plt.ylabel("ΔΣ [M$_\\odot$/kpc$^2$]")
    plt.title(title)
    plt.legend(fontsize=8)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def main():
    kids_csv  = "data/raw/kids/KiDS_binB.csv"
    route_csv = "data/processed/routeA/routeA_m0p18_binB.csv"
    pnl_csv   = "data/raw/Pnl_z0p286_HMcode.csv"
    z = 0.286

    # FREE-A
    R,D,E,DS1,DS2,M,parsF = fit_case(kids_csv, route_csv, pnl_csv, z, case="FREEA")
    pd.DataFrame({"R":R, "DS2":DS2}).to_csv("results/tables/binB_2halo_free.csv", index=False)
    pd.DataFrame({"R":R, "DS_total":M}).to_csv("results/tables/binB_total_free.csv", index=False)
    save_overlay(R,D,E,DS1,DS2,M,
                 f"KiDS B — PFGM free fit (χ²/ν≈{parsF['chi2_red']:.2f}, λ≈{parsF['lambda_mpc']:.3g} Mpc)",
                 "figures/lensing/overlay_binB_routeA_with_2halo_free.png")

    # SPARC-locked
    R,D,E,DS1,DS2,M,parsL = fit_case(kids_csv, route_csv, pnl_csv, z, case="A1")
    pd.DataFrame({"R":R, "DS2":DS2}).to_csv("results/tables/binB_2halo_SPARClocked.csv", index=False)
    pd.DataFrame({"R":R, "DS_total":M}).to_csv("results/tables/binB_total_SPARClocked.csv", index=False)
    save_overlay(R,D,E,DS1,DS2,M,
                 f"KiDS B — SPARC-locked (χ²/ν≈{parsL['chi2_red']:.2f}, λ≈{parsL['lambda_mpc']:.3g} Mpc)",
                 "figures/lensing/overlay_binB_routeA_with_2halo_SPARClocked.png")

    # GR kernel control
    R,D,E,DS1,DS2,M,parsG = fit_case(kids_csv, route_csv, pnl_csv, z, case="GR")
    pd.DataFrame({"R":R, "DS_total":M}).to_csv("results/tables/binB_total_GRkernel.csv", index=False)
    save_overlay(R,D,E,DS1,DS2,M,
                 f"KiDS B — GR kernel (λ=0; χ²/ν≈{parsG['chi2_red']:.2f})",
                 "figures/lensing/overlay_binB_GRkernel.png")

    # write a concise JSON fit summary (free & locked & GR)
    out = dict(
        free = dict(A1h=parsF["A1h"], b0=parsF["b0"], b2=parsF["b2"], lambda_mpc=parsF["lambda_mpc"], chi2_red=parsF["chi2_red"]),
        SPARClocked = dict(A1h=1.0, b0=parsL["b0"], b2=parsL["b2"], lambda_mpc=parsL["lambda_mpc"], chi2_red=parsL["chi2_red"]),
        GRkernel = dict(lambda_mpc=0.0, chi2_red=parsG["chi2_red"])
    )
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/tables/binB_fit_summary_from_repo.json","w"), indent=2)

    print("Done. Free:", parsF, "\nLocked:", parsL, "\nGR:", parsG)

if __name__=="__main__":
    main()
