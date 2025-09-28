import os, io, re, zipfile
import numpy as np
import pandas as pd

IN_ZIP   = "data/raw/sparc/Rotmod_LTG.zip"
OUT_DIR  = "data/raw/sparc/curves"

# Column name candidates (lowercased, stripped)
R_KEYS      = ["r_kpc","r (kpc)","radius_kpc","radius (kpc)","r","radius"]
VOBS_KEYS   = ["vobs","v_obs","vrot","vrot_kms","v_rot","v_obs_kms"]
VDISK_KEYS  = ["vdisk","v_disk","vstars_disk","v_stars_disk","vd"]
VBULGE_KEYS = ["vbulge","v_bulge","vstars_bulge","v_stars_bulge","vb"]
VGAS_KEYS   = ["vgas","v_gas","vhi","v_hi","vneutral","v_neutral"]

def pick(df, keys):
    cols = {str(c).lower().strip(): c for c in df.columns}
    for k in keys:
        if k in cols: return cols[k]
    return None

def clean_numeric(s):
    # Remove units or weird strings like 'kpc' 'km/s'
    if isinstance(s, str):
        s = re.sub(r"[^\d\.\-eE+]", " ", s)
    return s

def read_table(raw):
    # Try flexible CSV read (delimiter sniffing; ignore comment lines)
    for hdr in [0, None]:
        try:
            df = pd.read_csv(io.StringIO(raw), header=hdr, sep=None, engine="python", comment='#')
            if df.shape[1] >= 2:
                # strip whitespace in header
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            continue
    # Fallback: whitespace split
    try:
        df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, comment='#', header=None)
        # create generic headers
        df.columns = [f"col{i}" for i in range(df.shape[1])]
        return df
    except Exception:
        return None

def to_curve(df):
    # Normalize numerics
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c].apply(clean_numeric), errors="ignore")
        except Exception:
            pass

    # Map columns by candidates (case-insensitive)
    cols_low = {str(c).lower().strip(): c for c in df.columns}
    def pick_col(keys):
        for k in keys:
            if k in cols_low:
                return cols_low[k]
        return None

    Rcol     = pick_col(R_KEYS)
    Vobs_col = pick_col(VOBS_KEYS)
    Vd_col   = pick_col(VDISK_KEYS)
    Vb_col   = pick_col(VBULGE_KEYS)
    Vg_col   = pick_col(VGAS_KEYS)

    # Must have R and at least baryonic components or vobs
    if Rcol is None:
        return None

    # Coerce numerics; drop non-finite
    R  = pd.to_numeric(df[Rcol], errors="coerce")
    Vobs = pd.to_numeric(df[Vobs_col], errors="coerce") if Vobs_col else None
    Vd   = pd.to_numeric(df[Vd_col],   errors="coerce") if Vd_col   else None
    Vb   = pd.to_numeric(df[Vb_col],   errors="coerce") if Vb_col   else None
    Vg   = pd.to_numeric(df[Vg_col],   errors="coerce") if Vg_col   else None

    m = R.notna()
    if Vobs is not None: m &= Vobs.notna()
    if Vd   is not None: m &= Vd.notna()
    if Vb   is not None: m &= Vb.notna()
    if Vg   is not None: m &= Vg.notna()
    dfc = pd.DataFrame({"R_kpc": R[m]})
    if dfc.empty: return None

    # Build baryonic velocity (sum in quadrature of available components)
    parts = []
    for comp in [Vd, Vb, Vg]:
        if comp is not None:
            parts.append(comp[m].to_numpy(dtype=float)**2)
    if len(parts)==0:
        return None
    Vbar = np.sqrt(np.sum(parts, axis=0))

    out = pd.DataFrame({"R_kpc": dfc["R_kpc"].to_numpy(dtype=float),
                        "V_baryon_kms": Vbar})

    # Observed velocity if present
    if Vobs is not None:
        out["V_obs_kms"] = Vobs[m].to_numpy(dtype=float)
    else:
        # If no observed column, skip; we need V_obs for the phenomenological residual fit
        return None

    # Sort by R and drop duplicates
    out = out.sort_values("R_kpc").drop_duplicates(subset="R_kpc")
    # Keep only finite, positive radii and velocities
    out = out[(out["R_kpc"]>0) & np.isfinite(out["V_baryon_kms"]) & np.isfinite(out["V_obs_kms"])]
    if len(out) < 6:
        return None
    return out[["R_kpc","V_obs_kms","V_baryon_kms"]]

def main():
    if not os.path.isfile(IN_ZIP):
        raise FileNotFoundError(IN_ZIP)
    os.makedirs(OUT_DIR, exist_ok=True)

    nz = 0; ok = 0; fail = 0
    with zipfile.ZipFile(IN_ZIP, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or not (name.lower().endswith('.txt') or name.lower().endswith('.csv') or name.lower().endswith('.dat')):
                continue
            nz += 1
            raw = zf.read(name).decode("utf-8","ignore")
            df = read_table(raw)
            if df is None:
                fail += 1
                continue
            curve = to_curve(df)
            if curve is None:
                fail += 1
                continue
            # Galaxy name from filename stem
            gal = os.path.splitext(os.path.basename(name))[0]
            # Tidy galaxy name (remove spaces that break filenames)
            gal = gal.replace(" ", "_")
            curve.insert(0, "Galaxy", gal)
            outp = os.path.join(OUT_DIR, f"{gal}.csv")
            curve.to_csv(outp, index=False)
            ok += 1
            print(f"Wrote {outp} ({len(curve)} rows)")

    print(f"Scanned {nz} files in ZIP: extracted {ok}, failed {fail}")
    print(f"Curves directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
