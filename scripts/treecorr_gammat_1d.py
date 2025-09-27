#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck15
import astropy.units as u
import treecorr

# ---------- helpers ----------
def _has(cols, name):
    try:
        return name in cols.names
    except Exception:
        return False

def _first_present(cols, candidates):
    for c in candidates:
        if _has(cols, c):
            return c
    return None

def load_sources(path, e1_col, e2_col, w_col, zb_col):
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        cols = hdul[1].columns
        # KiDS DR4.1 has ALPHA_J2000 / DELTA_J2000
        ra_name  = _first_present(cols, ["ALPHA_J2000", "RAJ2000", "RA"])
        dec_name = _first_present(cols, ["DELTA_J2000", "DECJ2000", "DEC"])
        if ra_name is None or dec_name is None:
            raise KeyError("Could not find RA/DEC columns in sources. Tried ALPHA_J2000/DELTA_J2000 and RAJ2000/DECJ2000 and RA/DEC.")
        ra  = np.asarray(d[ra_name], dtype=float)
        dec = np.asarray(d[dec_name], dtype=float)
        e1  = np.asarray(d[e1_col], dtype=float)
        e2  = np.asarray(d[e2_col], dtype=float)
        w   = np.asarray(d[w_col],  dtype=float)
        zb  = np.asarray(d[zb_col], dtype=float)
    return ra, dec, e1, e2, w, zb

def _compose_boss_weight(cols, d):
    """Compose a reasonable BOSS lens weight if present."""
    w = None
    # Common BOSS DR12 fields:
    # WEIGHT_FKP, WEIGHT_SYSTOT, WEIGHT_CP, WEIGHT_NOZ
    w_fkp = d["WEIGHT_FKP"]    if _has(cols, "WEIGHT_FKP")    else None
    w_sys = d["WEIGHT_SYSTOT"] if _has(cols, "WEIGHT_SYSTOT") else None
    w_cp  = d["WEIGHT_CP"]     if _has(cols, "WEIGHT_CP")     else None
    w_noz = d["WEIGHT_NOZ"]    if _has(cols, "WEIGHT_NOZ")    else None
    if w_fkp is not None:
        w = np.array(w_fkp, dtype=float)
        if w_sys is not None: w = w * np.array(w_sys, dtype=float)
        if w_cp  is not None: w = w * np.array(w_cp,  dtype=float)
        if w_noz is not None: w = w * (1.0 + np.array(w_noz, dtype=float))
    return w

def load_lenses_or_randoms(path, expect_z=True):
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        cols = hdul[1].columns
        # BOSS uses RA/DEC; some catalogs may have ALPHA_J2000/DELTA_J2000
        ra_name  = _first_present(cols, ["RA", "ALPHA_J2000", "RAJ2000"])
        dec_name = _first_present(cols, ["DEC", "DELTA_J2000", "DECJ2000"])
        if ra_name is None or dec_name is None:
            raise KeyError(f"Could not find RA/DEC columns in {path}. Tried RA/DEC and ALPHA_J2000/DELTA_J2000.")
        ra  = np.asarray(d[ra_name], dtype=float)
        dec = np.asarray(d[dec_name], dtype=float)
        zl  = np.asarray(d["Z"], dtype=float) if expect_z and _has(cols, "Z") else np.zeros_like(ra, dtype=float)

        w = _compose_boss_weight(cols, d)
        if w is None:
            # Fall back to unity weight
            w = np.ones_like(ra, dtype=float)
    return ra, dec, zl, w

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Stack γ_t(θ) with TreeCorr and write tidy CSV.")
    ap.add_argument("--sources", required=True, help="KiDS WL FITS (.fits or .fits.gz)")
    ap.add_argument("--lenses",  required=True, help="Lens catalog (.fits or .fits.gz)")
    ap.add_argument("--randoms", required=True, help="Random lens catalog (.fits or .fits.gz)")

    ap.add_argument("--e1-col", default="e1")
    ap.add_argument("--e2-col", default="e2")
    ap.add_argument("--w-col",  default="weight")
    ap.add_argument("--zb-col", default="Z_B")

    ap.add_argument("--zb-min", type=float, default=None)
    ap.add_argument("--zb-max", type=float, default=None)
    ap.add_argument("--zl-min", type=float, default=None)
    ap.add_argument("--zl-max", type=float, default=None)

    ap.add_argument("--theta-min", type=float, default=0.5, help="arcmin")
    ap.add_argument("--theta-max", type=float, default=60.0, help="arcmin")
    ap.add_argument("--nbins",     type=int,   default=15)
    ap.add_argument("--bin-type",  choices=["Log", "Linear"], default="Log", help="TreeCorr binning type")

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # sources
    ra_s, dec_s, e1_s, e2_s, w_s, zs = load_sources(args.sources, args.e1_col, args.e2_col, args.w_col, args.zb_col)
    msrc = np.ones_like(zs, dtype=bool)
    if args.zb_min is not None: msrc &= (zs >= args.zb_min)
    if args.zb_max is not None: msrc &= (zs <  args.zb_max)
    ra_s, dec_s, e1_s, e2_s, w_s = ra_s[msrc], dec_s[msrc], e1_s[msrc], e2_s[msrc], w_s[msrc]

    # lenses & randoms
    ra_l, dec_l, zl, w_l = load_lenses_or_randoms(args.lenses,  expect_z=True)
    ra_r, dec_r, zr, w_r = load_lenses_or_randoms(args.randoms, expect_z=False)

    mlens = np.ones_like(zl, dtype=bool)
    if args.zl_min is not None: mlens &= (zl >= args.zl_min)
    if args.zl_max is not None: mlens &= (zl <  args.zl_max)
    ra_l, dec_l, zl, w_l = ra_l[mlens], dec_l[mlens], zl[mlens], w_l[mlens]

    # catalogs
    cat_s = treecorr.Catalog(ra=ra_s, dec=dec_s, g1=e1_s, g2=e2_s, w=w_s, ra_units='deg', dec_units='deg')
    cat_l = treecorr.Catalog(ra=ra_l, dec=dec_l, w=w_l,          ra_units='deg', dec_units='deg')
    cat_r = treecorr.Catalog(ra=ra_r, dec=dec_r, w=w_r,          ra_units='deg', dec_units='deg')

    # correlations
    ng = treecorr.NGCorrelation(min_sep=args.theta_min, max_sep=args.theta_max, nbins=args.nbins,
                                sep_units='arcmin', bin_type=args.bin_type)
    rg = treecorr.NGCorrelation(min_sep=args.theta_min, max_sep=args.theta_max, nbins=args.nbins,
                                sep_units='arcmin', bin_type=args.bin_type)

    ng.process(cat_l, cat_s)
    rg.process(cat_r, cat_s)

    # estimator: subtract random signal
    gammat     = ng.xi - rg.xi
    gammat_err = np.sqrt(ng.varxi + rg.varxi)
    weight     = ng.weight + rg.weight

    # mean separation in arcmin (use meanr, already in sep_units)
    theta_arcmin = ng.meanr

    # effective lens z and radii for convenience
    zl_eff = np.average(zl, weights=w_l) if len(zl) else np.nan
    Dl     = Planck15.angular_diameter_distance(zl_eff).to(u.Mpc).value if np.isfinite(zl_eff) else np.nan
    theta_rad = (theta_arcmin/60.0) * (np.pi/180.0)
    R_phys = Dl * theta_rad if np.isfinite(Dl) else np.full_like(theta_arcmin, np.nan)
    R_com  = R_phys * (1.0 + zl_eff) if np.isfinite(zl_eff) else np.full_like(theta_arcmin, np.nan)

    out = pd.DataFrame({
        "theta_arcmin": theta_arcmin,
        "R_phys_Mpc":   R_phys,
        "R_com_Mpc":    R_com,
        "gammat":       gammat,
        "gammat_err":   gammat_err,
        "weight":       weight,
        "z_l_eff":      np.full_like(theta_arcmin, zl_eff, dtype=float),
        "n_pairs_D":    ng.npairs,
        "n_pairs_R":    rg.npairs,
    })
    out.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
