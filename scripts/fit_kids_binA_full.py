import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy.special import jv
import json, os

def hankel_j2(k, Pgm, R):
    # ∫ (k dk / 2π) Pgm J2(kR)
    out = []
    for r in R:
        J2 = jv(2, k*r)
        out.append(np.trapz(k*Pgm*J2/(2*np.pi), k))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids", required=True)        # CSV: R [Mpc], DS [Msun/kpc^2], err
    ap.add_argument("--pnl",  required=True)        # CSV: k [1/Mpc], P [Mpc^3]
    ap.add_argument("--route1h", required=True)     # CSV: R [Mpc], DS_1h [Msun/kpc^2] (Route-A template on same radii)
    ap.add_argument("--z", type=float, default=0.286)
    ap.add_argument("--outjson", default="results/tables/binA_fit_FREEA_J2_FULL.json")
    args = ap.parse_args()

    # data
    dat  = pd.read_csv(args.kids).to_numpy(float)
    R, D, E = dat[:,0], dat[:,1], dat[:,2]
    oneh = pd.read_csv(args.route1h).to_numpy(float)
    if not np.allclose(R, oneh[:,0], rtol=0, atol=1e-8):
        raise SystemExit("Route-A 1h radii do not match KiDS radii; align/interpolate first.")
    DS1 = oneh[:,1]  # Msun/kpc^2

    tbl  = pd.read_csv(args.pnl).to_numpy(float)
    k, P = tbl[:,0], tbl[:,1]   # 1/Mpc, Mpc^3

    # constants & conventions
    h=0.674; Om=0.315
    rho_c0 = 2.775e11*(h**2)     # Msun/Mpc^3
    rho_m  = Om*rho_c0

    # model builder with (1+z)^2 normalization on 2-halo (per your check)
    def model(params):
        A1h, b0, b2, lam = params
        mu  = 1.0/(1.0 + (k*lam)**2)
        Pgm = (b0 + b2*k*k) * mu * P
        DS2 = rho_m * hankel_j2(k, Pgm, R) / 1e6              # Msun/kpc^2
        DS2 *= (1.0 + args.z)**2
        return A1h*DS1 + DS2

    # residuals (diagonal chi^2)
    def resid(params):
        return (model(params) - D) / np.where(E>0,E,1.0)

    # initial guesses + bounds
    p0  = np.array([  1.0,  1.6, 10.0, 1.2 ])    # A1h, b0, b2 [Mpc^2], lam [Mpc]
    lo  = np.array([  0.0,  0.5,  0.0, 0.01])
    hi  = np.array([100.0,  5.0, 50.0, 10.0])

    fit = least_squares(resid, p0, bounds=(lo,hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
    A1h, b0, b2, lam = fit.x
    r   = resid(fit.x)
    chi2 = float(np.sum(r*r))
    N    = len(R)
    kpar = 4
    AIC  = chi2 + 2*kpar
    BIC  = chi2 + kpar*np.log(N)

    os.makedirs(Path(args.outjson).parent, exist_ok=True)
    out = dict(mode="FREEA_J2_FULL_(1+z)^2", A1h=A1h, b0=b0, b2=b2, lambda_mpc=lam,
               chi2_red=chi2/(N-kpar), AIC=AIC, BIC=BIC, note="Diagonal fit with (1+z)^2 on 2h; exact J2 Hankel.")
    json.dump(out, open(args.outjson,"w"), indent=2)
    print("Fit:", out)

if __name__=="__main__":
    main()
