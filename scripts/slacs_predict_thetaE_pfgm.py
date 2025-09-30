import numpy as np, pandas as pd, os, re
from pathlib import Path

# crude cosmology distances (flat LCDM proxy for scaling); if you have a distance file, swap this
def Ez(z): return np.sqrt(0.3*(1+z)**3 + 0.7)
def Dc(z, nz=1000):
    zs = np.linspace(0, z, nz+1)
    dz = z/nz
    return 2997.92458*np.sum(1.0/Ez(zs))*dz  # Mpc/h-ish proxy; for ratios only

def D_ang(z1, z2=None):
    if z2 is None:
        return Dc(z)/(1+z)
    return (Dc(z2)-Dc(z1))/(1+z2)

def main():
    comp = Path("data/raw/slacs_compare_autogen.csv")
    if not comp.is_file():
        raise SystemExit("Missing data/raw/slacs_compare_autogen.csv")
    df = pd.read_csv(comp)
    # try to get zd, zs if present in your catalog (optional)
    cat = Path("data/raw/slacs/SLACS_table.cat")
    zd = zs = None
    if cat.is_file():
        try:
            raw = pd.read_csv(cat, sep=None, engine="python")
            raw.columns = [c.strip() for c in raw.columns]
            key = "#"
            if key not in raw.columns:
                key = next((c for c in raw.columns if re.search(r'(lens|name|id)', c, re.I)), raw.columns[0])
            raw["lens_key"] = raw[key].astype(str).str.strip().str.upper().str.replace(r'[\s:\-]+','', regex=True)
            df["lens_key"]  = df["lens"].astype(str).str.strip().str.upper()
            m = df.merge(raw, on="lens_key", how="left", suffixes=("","_raw"))
            zd = m["zd"] if "zd" in m.columns else None
            zs = m["zs"] if "zs" in m.columns else None
        except Exception:
            pass

    # very crude sigma_v proxy from GR θE (invert SIS under GR) as a neutral baseline
    # then apply a mild rescaling tied to λ posterior mass-scale; here we keep it neutral to avoid over-claiming
    th_gr = pd.to_numeric(df["thetaE_GR"], errors="coerce")  # arcsec
    # adopt sigma_v from GR thetaE to keep scale realistic: sigma_v^2 ∝ thetaE * Ds/Dds
    if zd is not None and zs is not None and np.isfinite(zd).any() and np.isfinite(zs).any():
        Dds = D_ang(zd, zs); Ds = D_ang(zs)
        ratio = np.where((Dds>0)&(Ds>0), Ds/Dds, np.nan)
    else:
        ratio = np.full_like(th_gr, np.nan, dtype=float)
    # constant factor for SIS in arcsec units absorbed; we keep proportional rescaling only
    sig2 = np.where(np.isfinite(th_gr)&np.isfinite(ratio), th_gr*ratio, np.nan)
    # PFGM predictor: neutral (same scale) with wide uncertainty until full calc is in
    thetaE_pfgm = th_gr.copy()
    sigma_theta = 0.3*np.abs(thetaE_pfgm)  # 30% wide

    out = pd.DataFrame(dict(
        lens=df["lens"],
        thetaE_GR=df["thetaE_GR"],
        thetaE_PFGM=thetaE_pfgm,
        sigma_GR=df.get("sigma_GR", np.nan),
        sigma_PFGM=sigma_theta
    ))
    out.to_csv("data/raw/slacs_compare_autogen_pfgmfilled.csv", index=False)
    print("Wrote data/raw/slacs_compare_autogen_pfgmfilled.csv with", len(out), "rows")
    print("NOTE: predictor is neutral (thetaE_PFGM≈thetaE_GR) with 30% σ; replace with full PFGM calc when ready.")
