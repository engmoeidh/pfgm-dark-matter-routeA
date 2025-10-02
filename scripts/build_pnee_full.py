#!/usr/bin/env python
import argparse, numpy as np, pandas as pd
from astropy.io import fits

# Cosmology helpers
def Ez(z, Om, Ol): return np.sqrt(Om*(1+z)**3 + Ol)
def comoving_chi(z,h,Om,Ol):
    c=299792.458; H0=100*h
    zz=np.linspace(0,z,4001); Ezv=Ez(zz,Om,Ol)
    return (c/H0)*np.trapz(1.0/Ezv, zz)
def DA(z,h,Om,Ol): return 0.0 if z<=0 else comoving_chi(z,h,Om,Ol)/(1+z)
def Hz(z,h,Om,Ol): H0=100*h; return H0*Ez(z,Om,Ol) # km/s/Mpc

def normalize_nz(z, nz):
    z = np.array(z,float); nz=np.clip(np.array(nz,float),0,np.inf)
    I = np.trapz(nz, z); 
    if I<=0: raise ValueError("n(z) normalization integral is zero.")
    return z, nz/I

def load_source_nz_from_fits(fits_path):
    hd = fits.open(fits_path)
    src = None
    for hdu in hd:
        if hdu.name.upper().startswith("NZ_SOURCE"):
            src=hdu; break
    if src is None: 
        hd.info(); raise SystemExit("NZ_SOURCE HDU not found.")
    cols = src.data.columns.names
    # choose z-mid; sum all bins
    z = np.asarray(src.data["Z_MID" if "Z_MID" in cols else "Z_LOW"], float).ravel()
    nsum = None
    for name in cols:
        if name.upper().startswith("BIN"):
            v = np.asarray(src.data[name], float).ravel()
            nsum = v if nsum is None else (nsum+v)
    if nsum is None: raise SystemExit("Could not find BIN* columns in NZ_SOURCE.")
    return normalize_nz(z, nsum)

def load_lens_nz_txt(path):
    arr=np.loadtxt(path)
    if arr.ndim!=2 or arr.shape[1]<2: raise ValueError(f"Bad lens n(z) txt {path}")
    return normalize_nz(arr[:,0], arr[:,1])

def ell_edges_from_ang_centers(ang_arcmin):
    # ang centers (arcmin) -> theta centers (rad)
    th = np.array(ang_arcmin,float)*(np.pi/180.0)/60.0
    th = th[np.argsort(th)]
    # theta edges at midpoints in log-space
    # build interior edges as geometric mean; extend outer edges by same ratio
    gmid = np.sqrt(th[:-1]*th[1:])
    # extend bounds
    r_lo = th[0]**2 / gmid[0]
    r_hi = th[-1]**2 / gmid[-1]
    th_edges = np.concatenate([[r_lo], gmid, [r_hi]])
    ell_edges = 2.0*np.pi/np.clip(th_edges,1e-12,None)
    # ensure ascending
    ell_edges = np.sort(ell_edges)
    return ell_edges

def build_Cell_gk(h,Om,Ol, zgrid, nzl, nzs, k_arr, P_arr, b0,b2, lam, ell):
    # Limber: C_ell = ∫ dz [ H(z)/c * Wg(z) Wk(z) / chi(z)^2 ] Pgk(k,z)
    c=299792.458
    # Distances & weights on zgrid
    chi = np.array([comoving_chi(z,h,Om,Ol) for z in zgrid])
    Hzv = Hz(zgrid,h,Om,Ol)
    # Source lensing kernel W_kappa(z)
    # First precompute g(z) = ∫_z^∞ dzs n_s(zs) (chi_s - chi)/chi_s
    # numerical cumulative integral from high-z downwards
    nzs_norm = nzs/np.trapz(nzs, zgrid)
    chi_s = chi
    # prepare cumulative integral I(z) = ∫_z^∞ n_s(zs)*(chi_s-chi)/chi_s dzs
    I = np.zeros_like(zgrid)
    # integrate in reverse
    acc = 0.0
    for i in range(len(zgrid)-1, -1, -1):
        if i < len(zgrid)-1:
            dz = zgrid[i+1]-zgrid[i]
            acc += nzs_norm[i+1]*max(chi_s[i+1]-chi[i],0.0)/max(chi_s[i+1],1e-9)*dz
        I[i] = acc
    Wk = 1.5*Om*(100*h/c)**2 * (1+zgrid)*chi * I  # dimensionless

    # Galaxy kernel Wg(z) ∝ n_l(z) (already normalized)
    nzl_norm = nzl/np.trapz(nzl, zgrid)
    Wg = nzl_norm

    # now evaluate Pgm(k,z) with k=(ell+0.5)/chi
    ell = np.atleast_1d(ell).astype(float)
    Cl = np.zeros_like(ell)
    for j, L in enumerate(ell):
        k = (L+0.5)/np.clip(chi,1e-9,None)
        mu = 1.0/(1.0+(k*lam)**2) if lam>0 else np.ones_like(k)
        Pnl = np.interp(k, k_arr, P_arr, left=0.0, right=0.0)
        Pgm = (b0 + b2*k*k) * mu * Pnl
        integrand = (Hzv/c) * (Wg*Wk/(np.clip(chi,1e-9,None)**2)) * Pgm
        Cl[j] = np.trapz(integrand, zgrid)
    return Cl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--pnl",  required=True)
    ap.add_argument("--lens_nz", required=True)   # txt: z n(z) for lens sample (LOWZ or CMASS)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--Om", type=float, default=0.315)
    ap.add_argument("--Ol", type=float, default=0.685)
    ap.add_argument("--b0", type=float, required=True)
    ap.add_argument("--b2", type=float, required=True)
    ap.add_argument("--lam",type=float, required=True) # Mpc
    ap.add_argument("--zmin", type=float, default=0.01)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--nz",   type=int,   default=600)
    ap.add_argument("--data", default="data/raw/kids/official/KiDS_PneE_data_vector_71.txt")
    ap.add_argument("--cov",  default="data/raw/kids/official/KiDS_PneE_cov_71x71.txt")
    ap.add_argument("--out",  default="results/tables/KiDS_PneE_model_full_71.txt")
    args=ap.parse_args()

    # Source nz from FITS, lens nz from txt
    z_s, n_s = load_source_nz_from_fits(args.fits)
    z_l, n_l = load_lens_nz_txt(args.lens_nz)

    # Unified z grid for kernels
    zmin = max(args.zmin, min(z_l.min(), z_s.min()))
    zmax = min(args.zmax, max(z_l.max(), z_s.max()))
    zgrid = np.linspace(zmin, zmax, args.nz)
    # Interpolate n(z) onto zgrid
    n_sg = np.interp(zgrid, z_s, n_s, left=0.0, right=0.0)
    n_lg = np.interp(zgrid, z_l, n_l, left=0.0, right=0.0)

    # Band edges from ANG centers
    hd = fits.open(args.fits)
    pne=[hdu for hdu in hd if hdu.name.upper().startswith("PNEE")][0]
    ang_arcmin = np.asarray(pne.data["ANG"], float).ravel()
    ell_edges  = ell_edges_from_ang_centers(ang_arcmin)

    # Pnl at z~0.286 (thin-lens approx)
    kp = pd.read_csv(args.pnl).to_numpy(float); k_arr, P_arr = kp[:,0], kp[:,1]

    # Evaluate Cl per band (average over ell in each band)
    Cl_bands = []
    for b in range(len(ell_edges)-1):
        lo, hi = ell_edges[b], ell_edges[b+1]
        ell = np.linspace(lo, hi, 64)
        Cl = build_Cell_gk(args.h, args.Om, args.Ol, zgrid, n_lg, n_sg, k_arr, P_arr, args.b0, args.b2, args.lam, ell)
        Cl_bands.append(Cl.mean())
    model = np.array(Cl_bands, float)

    # Align to official 71-bin PneE
    d = np.loadtxt(args.data); C = np.loadtxt(args.cov)
    n = min(len(model), len(d), C.shape[0], C.shape[1])
    m, d, C = model[:n], d[:n], C[:n,:n]
    w = np.linalg.eigvalsh(C)
    if w.min()<=0: C = C + (1e-12*np.median(np.diag(C)))*np.eye(n)
    Ci = np.linalg.inv(C)
    r = (m - d).reshape(-1,1)
    chi2 = float((r.T @ Ci @ r)[0,0])

    import pathlib
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(outp, m)
    print(f"[PneE full] wrote={outp}  n={n}  chi2={chi2:.6f}  chi2/nu={chi2/n:.6f}")
    for i in range(min(3,n)):
        print(f"  [{i}] model={m[i]:.4e}  data={d[i]:.4e}")

if __name__=="__main__":
    main()