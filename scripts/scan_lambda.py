import os, yaml, json
import numpy as np
import pandas as pd
from scipy import optimize
from src.pfgm.kernels import mu_lens, hankel_j2
from scripts.batch_fit_bins import load_kids_clean, load_routeA_1h, build_pnl

def chi2(model, data, cov=None, sigma=None):
    r = model - data
    if cov is not None:
        try:
            inv = np.linalg.inv(cov)
            return float(r @ inv @ r)
        except Exception:
            pass
    return float(np.sum((r/sigma)**2))

def fit_at_lambda(man, bin_sel, lam, use_cov=True):
    # paths & data
    kids, oneh_path, cov_path = None, None, None
    bin_sel = bin_sel.upper()
    if bin_sel=="C":
        kids = man["paths"]["kids_binC_clean"]
        oneh_path = man["paths"]["routeA_1h_binC"]
        cov_path = man["paths"].get("kids_binC_cov")
    elif bin_sel=="A":
        kids = man["paths"]["kids_binA_clean"]
        oneh_path = man["paths"]["routeA_1h_binA"]
        cov_path = man["paths"].get("kids_binA_cov")
    elif bin_sel=="B":
        kids = man["paths"]["kids_binB_clean"]
        oneh_path = man["paths"]["routeA_1h_binB"]
        cov_path = man["paths"].get("kids_binB_cov")
    else:
        raise ValueError("bin must be A/B/C")

    R, DS, S, C = load_kids_clean(kids, cov_path if use_cov else None)
    oneh        = load_routeA_1h(oneh_path, R)
    k, Pnl      = build_pnl(man["paths"]["p_nl"],
                            man["fitting"]["k_min"], man["fitting"]["k_max"],
                            man["fitting"]["n_k"])
    # inner fit for [A1h, b0, b2] at fixed lam
    x0  = np.array([1.0, 1.0, 1.0])  # mild prior
    lb  = np.array([0.0, 0.2, 0.0])
    ub  = np.array([2e3, 5.0, 50.0])

    def residual(z):
        A1h, b0, b2 = z
        twoh = hankel_j2(k, (b0 + b2*(k**2))*mu_lens(k, lam)*Pnl, R)
        mod  = A1h*oneh + twoh
        if C is not None:
            try:
                L = np.linalg.cholesky(C); return np.linalg.solve(L, mod-DS)
            except Exception:
                pass
        return (mod-DS)/S

    res = optimize.least_squares(residual, x0, bounds=(lb,ub),
                                 xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
    A1h,b0,b2 = res.x
    twoh = hankel_j2(k, (b0 + b2*(k**2))*mu_lens(k, lam)*Pnl, R)
    mod  = A1h*oneh + twoh
    c2   = chi2(mod, DS, cov=C, sigma=S)
    nu   = len(R) - len(res.x)
    return dict(lam=float(lam), chi2=float(c2), nu=int(nu),
                A1h=float(A1h), b0=float(b0), b2=float(b2))

def main():
    man = yaml.safe_load(open("configs/run_manifest.yaml","r"))
    lam_grid = np.geomspace(0.03, 10.0, 30)  # broader than your bounds
    out = []
    for lam in lam_grid:
        d = fit_at_lambda(man, "C", lam, use_cov=False)  # set True if you have cov
        out.append(d)
        print(f"lam={lam:.3g} chi2={d['chi2']:.3f} A1h={d['A1h']:.3g} b0={d['b0']:.3g} b2={d['b2']:.3g}")
    df = pd.DataFrame(out)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/binC_scan_lambda.csv", index=False)
    print("Wrote results/tables/binC_scan_lambda.csv")

if __name__ == "__main__":
    main()
