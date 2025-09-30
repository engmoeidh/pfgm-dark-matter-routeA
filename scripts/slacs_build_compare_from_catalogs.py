import re, h5py, numpy as np, pandas as pd
from pathlib import Path
from astropy.io import fits

OUT = Path("data/raw/slacs_compare_autogen.csv")

# --- helpers ---
def norm_lens(s):
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = s.upper()
    # normalize common forms (SDSSJXXXX±XXXX with/without colon/hyphen)
    s = s.replace("SDSSJ", "SDSSJ")
    s = s.replace(":", "")
    s = s.replace("-", "")
    return s

def try_hdf5(path):
    if not Path(path).is_file(): return None
    with h5py.File(path, "r") as h:
        # collect possible arrays
        keys = []
        def walk(g,p=""):
            for k,v in g.items():
                if isinstance(v,h5py.Dataset):
                    keys.append((p+k, v[...]))
                else:
                    walk(v,p+k+"/")
        walk(h)
    # map likely fields
    data = {}
    for k,arr in keys:
        kl = k.lower()
        if re.search(r"(name|lens|id)$", kl) and arr.dtype.kind in "SU":
            data["lens"] = np.array(arr, dtype=str)
        if re.search(r"(theta.?e|einstein|b_sie)$", kl):
            data["thetaE_GR"] = np.array(arr, dtype=float).squeeze()
        if re.search(r"(sig|unc|err).*(theta.?e|einstein|b_sie)", kl):
            data["sigma_GR"] = np.array(arr, dtype=float).squeeze()
    if "lens" in data and "thetaE_GR" in data:
        n = len(data["thetaE_GR"])
        out = pd.DataFrame({
            "lens": [norm_lens(x) for x in data["lens"]],
            "thetaE_GR": data["thetaE_GR"].astype(float),
            "sigma_GR": data.get("sigma_GR", np.full(n, np.nan)).astype(float),
            "thetaE_PFGM": np.nan, "sigma_PFGM": np.nan
        })
        return out
    return None

def try_fits(path):
    if not Path(path).is_file(): return None
    with fits.open(path) as hdul:
        for h in hdul:
            if hasattr(h,"data") and h.data is not None:
                cols = [c.name.lower() for c in h.columns] if hasattr(h,"columns") else []
                df = pd.DataFrame(np.array(h.data).byteswap().newbyteorder())
                df.columns = [str(c).strip() for c in df.columns]
                # guess lenses & thetaE columns
                lens_col = next((c for c in df.columns if re.search(r"(lens|name|id)", c, re.I)), None)
                th_col   = next((c for c in df.columns if re.search(r"(theta.?e|einstein|b_sie)", c, re.I)), None)
                sig_col  = next((c for c in df.columns if re.search(r"(sig|unc|err).*(theta.?e|einstein|b_sie)", c, re.I)), None)
                if lens_col and th_col:
                    out = pd.DataFrame({
                        "lens": df[lens_col].astype(str).map(norm_lens),
                        "thetaE_GR": pd.to_numeric(df[th_col], errors="coerce"),
                        "sigma_GR":  pd.to_numeric(df[sig_col], errors="coerce") if sig_col else np.nan
                    })
                    out["thetaE_PFGM"] = np.nan
                    out["sigma_PFGM"] = np.nan
                    return out.dropna(subset=["thetaE_GR"], how="all")
    return None

def try_cat(path):
    if not Path(path).is_file(): return None
    for sep in (None, r"\s+", ","):
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            break
        except Exception:
            df=None
    if df is None or df.empty: return None
    df.columns=[str(c).strip() for c in df.columns]
    lens_col = next((c for c in df.columns if re.search(r"(lens|name|id)", c, re.I)), None)
    th_col   = next((c for c in df.columns if re.search(r"(theta.?e|einstein|b_sie)", c, re.I)), None)
    sig_col  = next((c for c in df.columns if re.search(r"(sig|unc|err).*(theta.?e|einstein|b_sie)", c, re.I)), None)
    if lens_col and th_col:
        out = pd.DataFrame({
            "lens": df[lens_col].astype(str).map(norm_lens),
            "thetaE_GR": pd.to_numeric(df[th_col], errors="coerce"),
            "sigma_GR":  pd.to_numeric(df[sig_col], errors="coerce") if sig_col else np.nan,
            "thetaE_PFGM": np.nan, "sigma_PFGM": np.nan
        })
        return out.dropna(subset=["thetaE_GR"], how="all")
    return None

def main():
    paths = [
        "data/raw/slacs/full_inference.hdf5",
        "data/raw/slacs/slonly_inference.hdf5",
        "data/raw/slacs/full_pp.hdf5",
        "data/raw/slacs/parent_sample.fits",
        "data/raw/slacs/SLACS_table.cat",
    ]
    for p in paths:
        for fn in (try_hdf5, try_fits, try_cat):
            out = fn(p)
            if out is not None and not out.empty:
                out = out.drop_duplicates(subset=["lens"])
                OUT.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(OUT, index=False)
                print(f"Wrote {OUT} from {p} with N={len(out)}")
                return
    raise SystemExit("Could not extract θE table from available SLACS files.")

if __name__ == "__main__":
    main()
