#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
import astropy.units as u
from astropy.io import fits

def sigma_crit_Msun_pc2(zl, zs):
    """
    Σ_crit(phys) = c^2 / (4πG) * D_s / (D_l D_ls)
    Returns array (per-source); caller can form an effective Σ_crit using weights.
    """
    G = 4.30091e-3   # pc (km/s)^2 / Msun
    c = 299792.458   # km/s
    D_s  = Planck15.angular_diameter_distance(zs).to(u.pc).value
    D_l  = Planck15.angular_diameter_distance(zl).to(u.pc).value
    D_ls = Planck15.angular_diameter_distance_z1z2(zl, zs).to(u.pc).value
    sc = (c*c)/(4.0*np.pi*G) * (D_s/(D_l*D_ls))
    sc = np.asarray(sc, dtype=float)
    sc[zs <= zl] = np.inf
    return sc

def main():
    ap = argparse.ArgumentParser(description="γ_t → ΔΣ using effective Σ_crit from source Z_B.")
    ap.add_argument("--gammat",  required=True, help="CSV from treecorr_gammat_1d.py")
    ap.add_argument("--sources", required=True, help="KiDS WL FITS (to read Z_B and weights)")
    ap.add_argument("--zb-col",  default="Z_B")
    ap.add_argument("--w-col",   default="weight")
    ap.add_argument("--zl",      type=float, required=True, help="effective lens redshift")
    ap.add_argument("--out",     required=True)
    ap.add_argument("--comoving", action="store_true")
    args = ap.parse_args()

    # load source z & weights
    with fits.open(args.sources, memmap=True) as hdul:
        d = hdul[1].data
        cols = hdul[1].columns
        zs = np.asarray(d[args.zb_col], dtype=float)
        if args.w_col in cols.names:
            ws = np.asarray(d[args.w_col], dtype=float)
        else:
            ws = np.ones_like(zs, dtype=float)

    # effective Sigma_crit via harmonic mean with weights:
    #   Σ_eff^{-1} = < 1/Σ_i >_w  over sources with z_s > z_l
    sc_arr = sigma_crit_Msun_pc2(args.zl, zs)
    inv_sc = np.where(np.isfinite(sc_arr), 1.0/sc_arr, 0.0)
    sc_eff = 1.0 / np.average(inv_sc, weights=ws)

    df = pd.read_csv(args.gammat)
    if not {"gammat","gammat_err"}.issubset(df.columns):
        raise KeyError("Input CSV must have columns 'gammat' and 'gammat_err'.")

    ds_phys     = df["gammat"].to_numpy()      * sc_eff
    ds_phys_err = df["gammat_err"].to_numpy()  * sc_eff

    out = df.copy()
    out["SigmaCrit_eff_Msun_pc2"]          = sc_eff
    out["DeltaSigma_phys_Msun_per_pc2"]    = ds_phys
    out["DeltaSigma_phys_err_Msun_per_pc2"]= ds_phys_err

    if args.comoving:
        # ΔΣ_com = ΔΣ_phys / (1+z_l)^2 ; R_com already provided in the γ_t file (optional)
        fac = (1.0 + args.zl)**2
        out["DeltaSigma_comoving_Msun_per_pc2"]     = ds_phys * fac
        out["DeltaSigma_comoving_err_Msun_per_pc2"] = ds_phys_err * fac

    out.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
