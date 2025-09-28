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
    """
    Normalize a raw ROTMOD table (headered or headerless) into [R_kpc, V_obs_kms, V_baryon_kms].
    """

    import numpy as np
    import pandas as pd

    # --- Case 1: headerless .dat (columns named col0, col1, …) ---
    if all(str(c).startswith("col") for c in df.columns):
        cols = list(df.columns)
        # Typical SPARC order: R, Vobs, Verr, Vgas, Vdisk, Vbulge
        if len(cols) >= 6:
            df = df.rename(columns={
                cols[0]: "R_kpc",
                cols[1]: "V_obs_kms",
                cols[3]: "V_gas",
                cols[4]: "V_disk",
                cols[5]: "V_bulge",
            })
        elif len(cols) >= 5:
            df = df.rename(columns={
                cols[0]: "R_kpc",
                cols[1]: "V_obs_kms",
                cols[3]: "V_gas",
                cols[4]: "V_disk",
            })
        else:
            df = df.rename(columns={
                cols[0]: "R_kpc",
                cols[1]: "V_obs_kms",
            })

    # --- Try to locate columns by common names ---
    cols_low = {str(c).lower().strip(): c for c in df.columns}
    Rcol = None
    for k in ["r_kpc","radius","r"]:
        if k in cols_low: Rcol = cols_low[k]; break
    Vobs_col = None
    for k in ["vobs","v_obs","vrot","v_obs_kms"]:
        if k in cols_low: Vobs_col = cols_low[k]; break
    Vd_col = None
    for k in ["vdisk","v_disk"]: 
        if k in cols_low: Vd_col = cols_low[k]; break
    Vb_col = None
    for k in ["vbulge","v_bulge"]:
        if k in cols_low: Vb_col = cols_low[k]; break
    Vg_col = None
    for k in ["vgas","v_gas"]:
        if k in cols_low: Vg_col = cols_low[k]; break

    if Rcol is None or Vobs_col is None:
        return None

    # Extract numeric arrays
    R = pd.to_numeric(df[Rcol], errors="coerce")
    Vobs = pd.to_numeric(df[Vobs_col], errors="coerce")
    parts = []
    for col in [Vd_col, Vb_col, Vg_col]:
        if col is not None:
            parts.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)**2)

    if len(parts) == 0:
        return None

    Vbar = np.sqrt(np.sum(parts, axis=0))
    m = R.notna() & Vobs.notna() & np.isfinite(Vbar)
    out = pd.DataFrame({
        "R_kpc": R[m].to_numpy(dtype=float),
        "V_obs_kms": Vobs[m].to_numpy(dtype=float),
        "V_baryon_kms": Vbar[m],
    })
    out = out.sort_values("R_kpc").drop_duplicates(subset="R_kpc")
    return out if len(out) >= 6 else None


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

    # Fallback for headerless ROTMOD tables:
    if df is not None and all(str(c).startswith("col") for c in df.columns):
        # Typical SPARC ROTMOD order: R[kpc], Vobs, Verr, Vgas, Vdisk, Vbulge
        cols = list(df.columns)
        mapping = {}
        if len(cols) >= 6:
            mapping = {cols[0]:"R_kpc", cols[1]:"Vobs", cols[3]:"Vgas", cols[4]:"Vdisk", cols[5]:"Vbulge"}
        elif len(cols) >= 5:
            mapping = {cols[0]:"R_kpc", cols[1]:"Vobs", cols[3]:"Vgas", cols[4]:"Vdisk"}
        else:
            mapping = {cols[0]:"R_kpc", cols[1]:"Vobs"}
        df = df.rename(columns=mapping)

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
