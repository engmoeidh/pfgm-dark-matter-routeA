import argparse, os, glob, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_vflat(path="data/raw/sparc/SPARC_Vflat_from_rotmod.csv"):
    df = pd.read_csv(path)
    # robust normalize
    gcol = "Galaxy" if "Galaxy" in df.columns else ( "galaxy" if "galaxy" in df.columns else df.columns[0] )
    if gcol != "Galaxy": df = df.rename(columns={gcol:"Galaxy"})
    vcols = [c for c in df.columns if re.search(r'v[_\s-]*flat', c, re.I)]
    if not vcols: raise FileNotFoundError("Vflat column not found in SPARC_Vflat_from_rotmod.csv")
    return df[["Galaxy", vcols[0]]].rename(columns={vcols[0]:"Vflat_kms"})

def norm_gname(s):
    s = str(s).strip()
    s = re.sub(r'_rotmod$','', s, flags=re.I)
    s = re.sub(r'\s+',' ', s)
    return s

def load_curves(curves_dir="data/raw/sparc/curves"):
    rows=[]
    for p in glob.glob(str(Path(curves_dir)/"*.csv")):
        df = pd.read_csv(p)
        cols = {c.lower():c for c in df.columns}
        need = ["r_kpc","v_obs_kms"]
        if not all(k in cols for k in need): continue
        R = df[cols["r_kpc"]].to_numpy(float)
        Vobs = df[cols["v_obs_kms"]].to_numpy(float)
        Vd = df[cols.get("v_disk_kms","")].to_numpy(float) if "v_disk_kms" in cols else np.zeros_like(R)
        Vb = df[cols.get("v_bulge_kms","")].to_numpy(float) if "v_bulge_kms" in cols else np.zeros_like(R)
        Vg = df[cols.get("v_gas_kms","")].to_numpy(float) if "v_gas_kms" in cols else np.zeros_like(R)
        gal = Path(p).stem
        rows.append(dict(Galaxy=gal,R_kpc=R,Vobs=Vobs,Vd=Vd,Vb=Vb,Vg=Vg))
    return rows

def running_median(x, y, nbins=25):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x,y = x[m], y[m]
    if len(x)<10: return (np.array([]),)*4
    qx = np.quantile(x, np.linspace(0,1,nbins+1))
    xc, ym, ylo, yhi = [], [], [], []
    for i in range(nbins):
        mask = (x>=qx[i]) & (x<qx[i+1]) if i<nbins-1 else (x>=qx[i])&(x<=qx[i+1])
        yy = y[mask]
        if len(yy)>=5:
            xc.append(np.median([qx[i],qx[i+1]]))
            ym.append(np.median(yy))
            ylo.append(np.quantile(yy, 0.16))
            yhi.append(np.quantile(yy, 0.84))
    return np.array(xc), np.array(ym), np.array(ylo), np.array(yhi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="results/tables/per_galaxy_master.csv")
    ap.add_argument("--vflat",  default="data/raw/sparc/SPARC_Vflat_from_rotmod.csv")
    ap.add_argument("--curves", default="data/raw/sparc/curves")
    ap.add_argument("--btfr",   default="figures/btfr/BTFR_MAIN_FINAL.png")
    ap.add_argument("--rar",    default="figures/rar/RAR_MAIN_FINAL.png")
    args = ap.parse_args()

    os.makedirs(Path(args.btfr).parent, exist_ok=True)
    os.makedirs(Path(args.rar).parent,  exist_ok=True)

    pm = pd.read_csv(args.master)
    pm["Galaxy_norm"] = pm["Galaxy"].map(norm_gname)

    vflat = load_vflat(args.vflat)
    vflat["Galaxy_norm"] = vflat["Galaxy"].map(norm_gname)
    j = pm.merge(vflat[["Galaxy_norm","Vflat_kms"]], on="Galaxy_norm", how="left")

    # --- BTFR (proxy Mb if absent) ---
    if "Mb" in j.columns:
        Mb = j["Mb"].to_numpy(float)
    else:
        # proxy Mb via Vflat^4 scaled to median mass ~ 10^10 Msun
        V = np.clip(j["Vflat_kms"].to_numpy(float), 1e-3, None)
        Mb = (V/200.0)**4 * 1e10  # Msun proxy
    V = np.clip(j["Vflat_kms"].to_numpy(float), 1e-3, None)

    plt.figure(figsize=(6.6,4.8))
    m = np.isfinite(Mb) & np.isfinite(V)
    plt.scatter(np.log10(V[m]), np.log10(Mb[m]), s=14, alpha=0.6, edgecolor="none")
    xs = np.linspace(np.nanmin(np.log10(V[m])), np.nanmax(np.log10(V[m])), 2)
    plt.plot(xs, 4*xs + (10 - 4*np.log10(200.0)), lw=1, alpha=0.9) # slope ~4 guide
    plt.xlabel(r"$\log_{10}\,V_{\rm flat}\ [{\rm km\,s^{-1}}]$")
    plt.ylabel(r"$\log_{10}\,M_{\rm b}\ [{\rm M_\odot}]\ (\mathrm{proxy})$")
    plt.title("BTFR (proxy Mb; replace with true Mb when available)")
    plt.tight_layout(); plt.savefig(args.btfr, dpi=160); plt.close()

    # --- RAR from curves ---
    clouds = load_curves(args.curves)
    gx, gy = [], []
    for rec in clouds:
        Rk = np.clip(rec["R_kpc"], 1e-3, None)
        # convert to consistent acceleration units (km/s)^2 per kpc; log-log slope is unit invariant
        gbar = (rec["Vd"]**2 + rec["Vb"]**2 + rec["Vg"]**2)/Rk
        gtot = (rec["Vobs"]**2)/Rk
        gx.append(np.log10(np.clip(gbar, 1e-12, None)))
        gy.append(np.log10(np.clip(gtot, 1e-12, None)))
    if len(gx):
        gx = np.concatenate(gx); gy = np.concatenate(gy)
        xc, ym, lo, hi = running_median(gx, gy, nbins=25)
        plt.figure(figsize=(6.6,4.8))
        plt.hexbin(gx, gy, gridsize=60, bins="log")
        if len(xc):
            plt.plot(xc, ym, color="white", lw=2)
            plt.fill_between(xc, lo, hi, color="white", alpha=0.2)
        xs = np.linspace(np.nanmin(gx), np.nanmax(gx), 2)
        plt.plot(xs, xs, lw=1, color="k", alpha=0.8)
        plt.xlabel(r"$\log_{10}\,g_{\rm bar}$ (km$^2$ s$^{-2}$ kpc$^{-1}$)")
        plt.ylabel(r"$\log_{10}\,g_{\rm tot}$ (km$^2$ s$^{-2}$ kpc$^{-1}$)")
        plt.title("RAR (from ROTMOD components)")
        plt.tight_layout(); plt.savefig(args.rar, dpi=160); plt.close()
        print("RAR cloud points:", len(gx))
    else:
        print("No SPARC curve CSVs found under", args.curves)

if __name__ == "__main__":
    main()
