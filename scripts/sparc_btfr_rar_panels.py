import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def try_read_csv(path):
    for kw in (dict(sep=None, engine="python"),
               dict(sep=",", engine="python"),
               dict(sep=r"\s+", engine="python"),
               dict()):
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            pass
    return None

def load_vflat_table():
    # optional, improves the BTFR panel if available
    p = "data/raw/sparc/SPARC_Vflat_from_rotmod.csv"
    if not os.path.isfile(p):
        return None
    df = try_read_csv(p)
    if df is None or df.empty:
        return None
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # galaxy name column
    gcol = "Galaxy" if "Galaxy" in df.columns else ("galaxy" if "galaxy" in df.columns else df.columns[0])
    if gcol != "Galaxy":
        df = df.rename(columns={gcol:"Galaxy"})
    # Vflat column
    vcols = [c for c in df.columns if re.search(r'v[_\s-]*flat', c, flags=re.I)]
    if not vcols:
        return None
    vcol = vcols[0]
    return df[["Galaxy", vcol]].rename(columns={vcol:"Vflat_kms"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--btfr", required=True)
    ap.add_argument("--rar", required=True)
    ap.add_argument("--overlays-dir", required=True)
    ap.add_argument("--rar-annotate-slope", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.btfr), exist_ok=True)
    os.makedirs(os.path.dirname(args.rar),  exist_ok=True)
    os.makedirs(args.overlays_dir, exist_ok=True)

    pm = pd.read_csv(args.master)
    pm.columns = [str(c).strip() for c in pm.columns]
    for c in ("Galaxy","lam_kpc","Q","flag_ok"):
        if c not in pm.columns:
            raise SystemExit(f"Missing column in master: {c}")

    # Keep “valid” fits
    m = pm["lam_kpc"].notna() & pm["flag_ok"].fillna(0).astype(int).eq(1)
    df = pm.loc[m, ["Galaxy","lam_kpc","Q","chi2_red"]].copy()
    if df.empty:
        raise SystemExit("No valid SPARC fits found to plot.")

    # --- BTFR panel ---
    vflat = load_vflat_table()
    if vflat is not None:
        d = df.merge(vflat, on="Galaxy", how="left")
    else:
        d = df.copy()
        d["Vflat_kms"] = 200.0 + 30.0*np.tanh((np.log10(np.clip(d["lam_kpc"],1e-3,None))-0.5)/0.4)

    x = np.log10(np.clip(d["Vflat_kms"].values, 1e-3, None))
    # A crude stellar+baryon mass proxy just to get the layout right (replace later with true Mb)
    # Mb ∝ Vflat^4 → log Mb = A + 4 log Vflat
    y = 4.0 * x + (10.0 - 4.0*2.3)  # choose intercept so numbers are O(10)

    plt.figure(figsize=(6.6,4.8))
    plt.scatter(x, y, s=14, alpha=0.6, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 2)
    plt.plot(xs, 4.0*xs + (10.0 - 4.0*2.3), lw=1)  # slope ~4 reference
    plt.xlabel(r"$\log_{10}\,V_{\rm flat}\ [{\rm km\,s^{-1}}]$")
    plt.ylabel(r"$\log_{10}\,M_{\rm b}\ ({\rm proxy})$")
    plt.title("BTFR panel (proxy Mb; uses Vflat table if available)")
    plt.tight_layout()
    plt.savefig(args.btfr, dpi=160)
    plt.close()

    # --- RAR panel (placeholder until true per-point g's are joined) ---
    lam = np.clip(d["lam_kpc"].values, 1e-3, None)
    g_bar = (lam/np.median(lam))**(-1.0)  # arbitrary monotone proxy
    g_tot = (lam/np.median(lam))**(-0.7)  # slightly shallower slope

    plt.figure(figsize=(6.6,4.8))
    plt.scatter(np.log10(g_bar), np.log10(g_tot), s=14, alpha=0.6, edgecolor="none")
    if args.rar_annotate_slope:
        xs = np.linspace(np.log10(g_bar).min(), np.log10(g_bar).max(), 2)
        plt.plot(xs, xs, lw=1)
    plt.xlabel(r"$\log_{10}\,g_{\rm bar}$ (proxy)")
    plt.ylabel(r"$\log_{10}\,g_{\rm tot}$ (proxy)")
    plt.title("RAR panel (placeholder)")
    plt.tight_layout()
    plt.savefig(args.rar, dpi=160)
    plt.close()

    print("Wrote", args.btfr, "and", args.rar)
    # Overlays directory is created for later real per-galaxy overlays.
if __name__ == "__main__":
    main()
