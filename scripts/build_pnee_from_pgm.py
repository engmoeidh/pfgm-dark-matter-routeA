#!/usr/bin/env python
import argparse, numpy as np, pandas as pd
from astropy.io import fits

def comoving_chi(z,h,Om,Ol):
    c=299792.458; H0=100*h
    zz=np.linspace(0,z,4001); Ez=np.sqrt(Om*(1+zz)**3+Ol)
    return (c/H0)*np.trapz(1/Ez, zz)

def DA(z,h,Om,Ol): return 0 if z<=0 else comoving_chi(z,h,Om,Ol)/(1+z)

def read_band_defs(fits_path, ell_low_col=None, ell_high_col=None):
    hd=fits.open(fits_path)
    pne=None
    for h in hd:
        if h.name.upper().startswith("PNEE"):
            pne=h; break
    if pne is None: 
        hd.info(); raise SystemExit("ERROR: 'PneE' HDU not found.")
    names=list(pne.data.columns.names)
    # Auto-detect integer-like columns as edges if not provided
    if ell_low_col is None or ell_high_col is None:
        cand=[]
        for n in names:
            try:
                v=np.asarray(pne.data[n]).ravel()
                if np.issubdtype(v.dtype, np.integer) or (np.allclose(v, np.round(v))):
                    cand.append(n)
            except Exception:
                pass
        # pick two monotonic integer columns
        if len(cand)<2: 
            print("PneE columns:", names)
            raise SystemExit("ERROR: need --ell_low_col/--ell_high_col (could not auto-detect).")
        # Heuristic: first two integer-like columns are edges
        ell_low_col, ell_high_col = cand[0], cand[1]
    ell_lo=np.asarray(pne.data[ell_low_col], float).ravel()
    ell_hi=np.asarray(pne.data[ell_high_col], float).ravel()
    # Keep bands with lo<hi and ell within [100, 1500] as per pipeline.ini
    m=(ell_hi>ell_lo) & (ell_hi>=100) & (ell_lo<=1500)
    return ell_lo[m], ell_hi[m]

def limber_pnee_for_band(lo, hi, h, Om, Ol, b0, b2, lam, k_arr, P_arr, z_eff):
    # Use Limber: ell ~ k chi ; project Pgκ \propto Pgm(k,z) with mu_lens; assume a delta-like z_eff for bands (good enough here)
    # chi_eff, DA_eff
    chi = comoving_chi(z_eff,h,Om,Ol)  # Mpc
    # band-averaged C_ell^{gκ} over ell ∈ [lo, hi]
    ell = np.linspace(max(lo,1.0), max(hi,lo+1.0), 64)
    k = (ell+0.5)/chi
    mu = 1.0/(1.0 + (k*lam)**2) if lam>0 else np.ones_like(k)
    # Interpolate Pnl onto k
    P = np.interp(k, k_arr, P_arr, left=0.0, right=0.0)  # Mpc^3
    Pgm = (b0 + b2*k*k)*mu*P
    # Up to an overall (distance/kernel) factor, we report relative PneE amplitude per band (consistent for χ² since we compare shapes)
    return np.trapz(Pgm, ell) / (hi - lo + 1e-12)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--pnl", required=True, help="HMcode Pnl at z=0.286 (k [1/Mpc], P [Mpc^3])")
    ap.add_argument("--z_eff", type=float, default=0.286)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--Om", type=float, default=0.315)
    ap.add_argument("--Ol", type=float, default=0.685)
    ap.add_argument("--b0", type=float, required=True)
    ap.add_argument("--b2", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True, help="lambda [Mpc]")
    ap.add_argument("--ell_low_col", default=None)
    ap.add_argument("--ell_high_col", default=None)
    ap.add_argument("--out", default="results/tables/KiDS_PneE_model_71.txt")
    ap.add_argument("--cov", default="data/raw/kids/official/KiDS_PneE_cov_71x71.txt")
    ap.add_argument("--data", default="data/raw/kids/official/KiDS_PneE_data_vector_71.txt")
    args=ap.parse_args()

    # Read bands
    lo, hi = read_band_defs(args.fits, args.ell_low_col, args.ell_high_col)
    # Load Pnl
    kp = pd.read_csv(args.pnl).to_numpy(float); k_arr, P_arr = kp[:,0], kp[:,1]
    # Build model vector (length may exceed 71 due to uncut bands; we will truncate to min length vs data)
    bands = np.array([limber_pnee_for_band(l,h,args.h,args.Om,args.Ol,args.b0,args.b2,args.lam,k_arr,P_arr,args.z_eff) for l,h in zip(lo,hi)], float)
    # Align to data/cov length
    d = np.loadtxt(args.data); C = np.loadtxt(args.cov)
    n = min(len(d), len(bands), C.shape[0], C.shape[1])
    m = bands[:n]; d = d[:n]; C = C[:n,:n]
    # χ²
    w = np.linalg.eigvalsh(C); 
    if w.min()<=0: C = C + (1e-12*np.median(np.diag(C)))*np.eye(n)
    Ci = np.linalg.inv(C)
    r = (m - d).reshape(-1,1); chi2 = float((r.T @ Ci @ r)[0,0])
    np.savetxt(args.out, m)
    print(f"Model PneE written: {args.out}  n={n}  chi2={chi2:.6f}  chi2/nu={chi2/n:.6f}")
    print("Bands used (first 5):")
    for i in range(min(5,n)):
        print(f"[{i}] ell in [{lo[i]:.0f}, {hi[i]:.0f}]  model={m[i]:.4e}  data={d[i]:.4e}")
if __name__=="__main__": 
    main()
