import os, yaml, numpy as np, pandas as pd
import emcee
from scipy import interpolate
from src.pfgm.kernels import mu_lens, hankel_j2_fast
from scripts.batch_fit_bins import load_kids_clean, load_routeA_1h, build_pnl

def loglike(theta, R, DS, S, C, oneh, k, Pnl):
    A1h,b0,b2,lam = theta
    if not (0.0<=A1h<=2000. and 0.05<=lam<=5.0 and 0.2<=b0<=5.0 and 0.0<=b2<=50.0):
        return -np.inf
    twoh = hankel_j2_fast(k, (b0 + b2*(k**2))*mu_lens(k, lam)*Pnl, R)
    mod  = A1h*oneh + twoh
    r = mod-DS
    if C is not None:
        try:
            L = np.linalg.cholesky(C); y = np.linalg.solve(L, r)
            return -0.5*np.dot(y,y)
        except Exception:
            pass
    return -0.5*np.sum((r/S)**2)

def main():
    man = yaml.safe_load(open("configs/run_manifest.yaml","r"))
    R, DS, S, C = load_kids_clean(man["paths"]["kids_binA_clean"], man["paths"].get("kids_binA_cov"))
    oneh        = load_routeA_1h(man["paths"]["routeA_1h_binA"], R)
    k, Pnl      = build_pnl(man["paths"]["p_nl"], man["fitting"]["k_min"], man["fitting"]["k_max"], man["fitting"]["n_k"])
    # initialize walkers near least-squares solution (use manifest initial as center)
    p0 = np.array([man["fitting"]["initial"]["A1h"], man["fitting"]["initial"]["b0"], man["fitting"]["initial"]["b2"], man["fitting"]["initial"]["lam"]])
    nwalk, ndim = 48, 4
    pos = p0 + 1e-2*np.random.randn(nwalk, ndim)
    sampler = emcee.EnsembleSampler(nwalk, ndim, loglike, args=(R,DS,S,C,oneh,k,Pnl))
    sampler.run_mcmc(pos, 3000, progress=False)
    chain = sampler.get_chain(discard=500, thin=5, flat=True)
    os.makedirs("results/posteriors", exist_ok=True)
    np.save("results/posteriors/binA_chain.npy", chain)
    qs = np.percentile(chain, [16,50,84], axis=0)
    print("A1h:  ", qs[:,0]); print("b0: ", qs[:,1]); print("b2: ", qs[:,2]); print("lam:", qs[:,3])

if __name__ == "__main__":
    main()
