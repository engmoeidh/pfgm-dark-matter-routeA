import os, re, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline

def read_auto(path):
    for hdr in [0, None]:
        try:
            df = pd.read_csv(path, header=hdr, sep=None, engine="python")
            return df
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")

def twofile_1h_from_total(path_total):
    p = Path(path_total)
    name = p.name
    # infer sibling 2h name by replacing tokens
    candidates = [
        name.replace("total","2halo"),
        name.replace("_total","_2halo"),
    ]
    sib = None
    for c in candidates:
        q = p.with_name(c)
        if q.exists():
            sib = q; break
    if sib is None:
        raise FileNotFoundError(f"Could not find 2halo sibling for {p}")
    dfT = read_auto(p)
    df2 = read_auto(sib)
    # coerce first two columns to numeric and align
    def pick_xy(df):
        cols = list(df.columns)
        x = pd.to_numeric(df[cols[0]], errors="coerce")
        y = pd.to_numeric(df[cols[1]], errors="coerce") if len(cols)>1 else None
        m = x.notna() & y.notna()
        return x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float)
    RT, Tot = pick_xy(dfT)
    R2, H2  = pick_xy(df2)
    # interpolate 2h onto RT
    H2i = np.interp(RT, R2, H2)
    OneH = Tot - H2i
    return RT, OneH

def main():
    man = yaml.safe_load(open("configs/run_manifest.yaml","r"))
    path_total = man["paths"]["routeA_1h_binC"]
    if not os.path.isfile(path_total):
        raise FileNotFoundError(path_total)
    RT, OneH = twofile_1h_from_total(path_total)

    # KiDS Bin-C radii (cleaned)
    kidsC = "data/processed/lensing/DeltaSigma_binC_clean.csv"
    df = pd.read_csv(kidsC)
    Rk = df["R"].astype(float).values
    f = InterpolatedUnivariateSpline(RT, OneH, k=1, ext=1)
    OneHk = f(Rk)

    # quick sanity plot
    os.makedirs("figures/lensing", exist_ok=True)
    plt.figure(figsize=(6,4.5))
    plt.loglog(RT, np.abs(OneH)+1e-12, '-', label="1-halo (total-2h) raw")
    plt.loglog(Rk, np.abs(OneHk)+1e-12, 'o', ms=4, label="1-halo on KiDS R")
    plt.xlabel("R [Mpc]")
    plt.ylabel(r"$|\Delta\Sigma_{1h}|$ [M$_\odot$/kpc$^2$]")
    plt.title("Bin-C Route-A 1-halo sanity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/lensing/bin_C_1h_sanity.png", dpi=160)
    plt.close()
    print("Wrote figures/lensing/bin_C_1h_sanity.png")

if __name__ == "__main__":
    main()
