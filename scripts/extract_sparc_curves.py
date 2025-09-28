import os, io, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

IN_ZIP  = "data/raw/sparc/Rotmod_LTG.zip"
OUT_DIR = "data/raw/sparc/curves"

def parse_rotmod_text(raw: str):
    """
    Parse a SPARC ROTMOD text block.
    Expected header in comments:
      # Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
      # kpc  km/s  km/s  km/s  km/s   km/s  L/pc^2  L/pc^2
    Returns a DataFrame with numeric columns and canonical names, or None.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    # find header line that starts with "# Rad"
    hdr_idx = None
    for i, ln in enumerate(lines[:10]):  # header within first few lines
        if ln.lstrip().startswith("# Rad"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None

    # Column names from that header line (remove '#', split on whitespace)
    cols = lines[hdr_idx].lstrip("#").strip().split()
    # Data starts after the units line (hdr_idx+2)
    data_lines = []
    for ln in lines[hdr_idx+2:]:
        if ln.startswith("#"):
            continue
        data_lines.append(ln)

    if len(data_lines) < 6 or len(cols) < 2:
        return None

    # Build DataFrame from data lines
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s+", engine="python", header=None)
    # Some ROTMODs have fewer than 8 columns; pad names accordingly
    cols = cols[:df.shape[1]]
    df.columns = cols

    # Canonical mapping
    name_map = {c.lower(): c for c in df.columns}
    def get(col_key):
        # accept short keys
        for k in [col_key, col_key.capitalize(), col_key.upper()]:
            if k in df.columns: return df[k]
        # specific common names
        if col_key == "Rad":
            return df[name_map.get("rad")] if "rad" in name_map else None
        if col_key == "Vobs":
            for k in ["Vobs","Vrot","V_obs","VROT","Vrot"]:
                if k in df.columns: return df[k]
            return df[name_map.get("vobs")] if "vobs" in name_map else None
        return None

    R    = pd.to_numeric(get("Rad"),  errors="coerce") if "Rad" in df.columns or "rad" in name_map else None
    Vobs = pd.to_numeric(get("Vobs"), errors="coerce")
    Vgas   = pd.to_numeric(df[name_map.get("vgas")],   errors="coerce") if "vgas"   in name_map else None
    Vdisk  = pd.to_numeric(df[name_map.get("vdisk")],  errors="coerce") if "vdisk"  in name_map else None
    Vbulge = pd.to_numeric(df[name_map.get("vbul")],   errors="coerce") if "vbul"   in name_map else (
             pd.to_numeric(df[name_map.get("vbulge")], errors="coerce") if "vbulge" in name_map else None)

    if R is None or Vobs is None:
        return None

    # Build baryon in quadrature from whatever is present
    parts = []
    for comp in (Vgas, Vdisk, Vbulge):
        if comp is not None:
            parts.append(comp.to_numpy(dtype=float)**2)
    if len(parts) == 0:
        return None

    Vbar = np.sqrt(np.sum(parts, axis=0))
    m = R.notna() & Vobs.notna() & np.isfinite(Vbar)
    if m.sum() < 6:
        return None
    out = pd.DataFrame({
        "R_kpc":        R[m].to_numpy(dtype=float),
        "V_obs_kms":    Vobs[m].to_numpy(dtype=float),
        "V_baryon_kms": Vbar[m],
    }).sort_values("R_kpc").drop_duplicates(subset="R_kpc")
    return out if len(out) >= 6 else None

def main():
    if not os.path.isfile(IN_ZIP):
        raise FileNotFoundError(IN_ZIP)
    os.makedirs(OUT_DIR, exist_ok=True)

    ok = fail = total = 0
    with zipfile.ZipFile(IN_ZIP, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"): continue
            ext = name.lower().rsplit(".",1)[-1]
            if ext not in {"dat","txt","csv"}: continue
            total += 1
            raw = zf.read(name).decode("utf-8","ignore")
            df = parse_rotmod_text(raw)
            if df is None:
                fail += 1
                continue
            gal = Path(name).stem.replace(" ", "_")
            df.insert(0, "Galaxy", gal)
            outp = os.path.join(OUT_DIR, f"{gal}.csv")
            df.to_csv(outp, index=False)
            ok += 1
            print(f"Wrote {outp} ({len(df)} rows)")
    print(f"Scanned {total} files in ZIP: extracted {ok}, failed {fail}")
    print(f"Curves directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
