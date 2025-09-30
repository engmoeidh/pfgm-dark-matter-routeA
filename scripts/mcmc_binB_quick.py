import os, yaml, numpy as np, emcee, json
from src.pfgm.kernels import mu_lens, hankel_j2_fast
from scripts.batch_fit_bins import load_kids_clean, load_routeA_1h, build_pnl

def loglike(th, R, DS, S, C, oneh, k, Pnl):
    A1h,b0,b2,lam = th
    if not (0.0<=A1h<=2000. and 0.05<=lam<=5.0 and 0.2<=b0<=5.0 and 0.0<=b2<=50.0):
        return -np.inf
    twoh = hankel_j2_fast(k, (b0 + b2*(k**2))*mu_lens(k, lam)*Pnl, R)
    mod  = A1h*oneh + twoh
    r = mod-DS
    if C is not None:
        try:
            L = np.linalg.cholesky(C); y = np.linalg.solve(L, r); return -0.5*np.dot(y,y)
        except Exception: pass
    return -0.5*np.sum((r/S)**2)

def main():
    man = yaml.safe_load(open("configs/run_manifest.yaml"))
    # Bin-B inputs (KiDS B)
    R, DS, S, C = load_kids_clean(man["paths"]["kids_binB_clean"], man["paths"].get("kids_binB_cov"))
    oneh        = load_routeA_1h(man["paths"]["routeA_1h_binA"], R)   # reuse A 1-halo, we interpolate
    k, Pnl      = build_pnl(man["paths"]["p_nl"], man["fitting"]["k_min"], man["fitting"]["k_max"], man["fitting"]["n_k"])

    # start near least-squares (if present)
    p0 = np.array([1.0,1.6,10.0,0.12])
    fitB = "results/tables/bin_B_fit_freeA.json"
    if os.path.isfile(fitB):
        d=json.load(open(fitB)); p0 = np.array([d["A1h"], d["b0"], d["b2"], d["lam"]], float)

    nwalk, ndim, steps = 24, 4, 800
    pos = p0 + 1e-2*np.random.randn(nwalk, ndim)
    sam = emcee.EnsembleSampler(nwalk, ndim, loglike, args=(R,DS,S,C,oneh,k,Pnl))
    sam.run_mcmc(pos, steps, progress=False)
    chain = sam.get_chain(discard=200, thin=5, flat=True)
    os.makedirs("results/posteriors", exist_ok=True)
    np.save("results/posteriors/binB_chain_quick.npy", chain)
    qs = np.percentile(chain, [16,50,84], axis=0)
    out = {"A1h":{"p16":float(qs[0,0]),"p50":float(qs[1,0]),"p84":float(qs[2,0])},
           "b0":{"p16":float(qs[0,1]),"p50":float(qs[1,1]),"p84":float(qs[2,1])},
           "b2":{"p16":float(qs[0,2]),"p50":float(qs[1,2]),"p84":float(qs[2,2])},
           "lam":{"p16":float(qs[0,3]),"p50":float(qs[1,3]),"p84":float(qs[2,3])}}
    with open("results/tables/binB_mcmc_freeA.json","w") as f: json.dump(out,f,indent=2)
    print("Bin-B λ (p50 ± 1σ) ≈", out["lam"]["p50"], "+/-", 0.5*(out["lam"]["p84"]-out["lam"]["p16"]))

if __name__ == "__main__":
    main()
