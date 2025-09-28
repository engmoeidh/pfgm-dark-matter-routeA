import os, io, re, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

IN_ZIP  = "data/raw/sparc/Rotmod_LTG.zip"
OUT_DIR = "data/raw/sparc/curves"

def read_table(raw_text):
    """
    Try to read a ROTMOD table from text.
    1) flexible CSV (delimiter sniffing; ignore comments)
    2) fallback whitespace (no header)
    Returns a DataFrame or None.
    """
    # try CSV-like
    for hdr in [0, None]:
        try:
            df = pd.read_csv(io.StringIO(raw_text), header=hdr, sep=None, engine="python", comment="#")
            if df.shape[1] >= 2:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            pass
    # try whitespace
    try:
        df = pd.read_csv(io.StringIO(raw_text), sep=r"\s+", engine="python", comment="#", header=None)
        ncols = df.shape[1]
        df.columns = [f"col{i}" for i in range(ncols)]
        return df
    except Exception:
        return None


def to_curve(df):
    """
    Normalize raw ROTMOD DataFrame to columns:
      R_kpc, V_obs_kms, V_baryon_kms

    Strategy:
      • If headerless (col0, col1, ...): try several plausible SPARC layouts:
          Layout A: R, Vobs, Verr, Vgas, Vdisk, Vbulge
          Layout B: R, Vgas, Vdisk, Vbulge, Vobs
          Layout C: R, Vobs, Vgas, Vdisk, (Vbulge)
      • If headered: pick by common names.
      • Require V_obs and ≥1 of (gas, disk, bulge).
    """
    import numpy as np
    import pandas as pd

    def build_out(R, Vobs, comps):
        parts = []
        for c in comps:
            if c is not None:
                parts.append(pd.to_numeric(c, errors="coerce").to_numpy(dtype=float)**2)
        if len(parts) == 0:
            return None
        Vbar = np.sqrt(np.sum(parts, axis=0))
        Rn    = pd.to_numeric(R,    errors="coerce")
        Vobsn = pd.to_numeric(Vobs, errors="coerce")
        m = Rn.notna() & Vobsn.notna() & np.isfinite(Vbar)
        if m.sum() < 6:  # need at least a handful of points
            return None
        out = pd.DataFrame({
            "R_kpc":        Rn[m].to_numpy(dtype=float),
            "V_obs_kms":    Vobsn[m].to_numpy(dtype=float),
            "V_baryon_kms": Vbar[m],
        })
        out = out.sort_values("R_kpc").drop_duplicates(subset="R_kpc")
        return out if len(out) >= 6 else None

    # ---- Headerless case: columns named col* ----
    if all(str(c).startswith("col") for c in df.columns):
        cols = list(df.columns)
        n = len(cols)
        # Pull by index helper (returns None if missing)
        def col(i): return df[cols[i]] if i < n else None

        # Try Layout A: [0]=R, [1]=Vobs, [3]=Vgas, [4]=Vdisk, [5]=Vbulge
        out = build_out(col(0), col(1), [col(3), col(4), col(5)])
        if out is not None:
            return out

        # Try Layout B: [0]=R, [4]=Vobs (last), [1]=Vgas, [2]=Vdisk, [3]=Vbulge
        out = build_out(col(0), col(max(1, n-1)), [col(1), col(2), col(3)])
        if out is not None:
            return out

        # Try Layout C: [0]=R, [1]=Vobs, [2]=Vgas, [3]=Vdisk, [4]=Vbulge
        out = build_out(col(0), col(1), [col(2), col(3), col(4)])
        if out is not None:
            return out

        # If still failing but we have at least 5 cols, brute-force: treat the last column as Vobs, preceding three as baryons
        if n >= 5:
            out = build_out(col(0), col(n-1), [col(1), col(2), col(3)])
            if out is not None:
                return out

        return None

    # ---- Headered case: map by name (more stable) ----
    cols_low = {str(c).lower().strip(): c for c in df.columns}
    def pick(*keys):
        for k in keys:
            if k in cols_low: return cols_low[k]
        return None

    Rcol   = pick("r_kpc","radius_kpc","r","radius")
    Vobs   = pick("v_obs_kms","vobs","v_obs","vrot","v_rot")
    Vgas   = pick("v_gas","vgas","vhi","v_hi")
    Vdisk  = pick("v_disk","vdisk","vstars_disk","v_stars_disk")
    Vbulge = pick("v_bulge","vbulge","vstars_bulge","v_stars_bulge")

    if Rcol is None or Vobs is None:
        return None

    return build_out(df[Rcol], df[Vobs], [df[c] for c in [Vgas, Vdisk, Vbulge] if c is not None])


def main():
    if not os.path.isfile(IN_ZIP):
        raise FileNotFoundError(IN_ZIP)
    os.makedirs(OUT_DIR, exist_ok=True)

    ok = 0; fail = 0; total = 0
    with zipfile.ZipFile(IN_ZIP, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"): 
                continue
            ext = name.lower().rsplit(".",1)[-1]
            if ext not in {"dat","txt","csv"}:
                continue
            total += 1
            try:
                raw = zf.read(name).decode("utf-8","ignore")
            except Exception:
                try:
                    raw = zf.read(name).decode("latin-1","ignore")
                except Exception:
                    fail += 1
                    continue
            df = read_table(raw)
            if df is None:
                fail += 1; continue
            cur = to_curve(df)
            if cur is None:
                fail += 1; continue
            gal = Path(name).stem.replace(" ", "_")
            cur.insert(0, "Galaxy", gal)
            outp = os.path.join(OUT_DIR, f"{gal}.csv")
            cur.to_csv(outp, index=False)
            ok += 1
            print(f"Wrote {outp} ({len(cur)} rows)")
    print(f"Scanned {total} files in ZIP: extracted {ok}, failed {fail}")
    print(f"Curves directory: {OUT_DIR}")

if __name__ == "__main__":
    main()
