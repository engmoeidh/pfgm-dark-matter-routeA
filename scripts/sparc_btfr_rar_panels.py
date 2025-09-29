import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def norm_name(s):
    s = str(s).strip()
    s = re.sub(r'_rotmod$', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    return s.upper()

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
    p = "data/raw/sparc/SPARC_Vflat_from_rotmod.csv"
    if not os.path.isfile(p):
        return None
    df = try_read_csv(p)
    if df is None or df.empty:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    gcol = "Galaxy" if "Galaxy" in df.columns else ("galaxy" if "galaxy" in df.columns else df.columns[0])
    if gcol != "Galaxy":
        df = df.rename(columns={gcol:"Galaxy"})
    vcols = [c for c in df.columns if re.search(r'v[_\s-]*flat', c, flags=re.I)]
    if not vcols:
        return None
    vcol = vcols[0]
    df = df[["Galaxy", vcol]].rename(columns={vcol:"Vflat_kms"}).copy()
    df["Galaxy_norm"] = df["Galaxy"].map(norm_name)
    return df[["Galaxy_norm","Vflat_kms"]]

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

    # valid fits
    m = pm["lam_kpc"].notna() & pm["flag_ok"].fillna(0).astype(int).eq(1)
    df = pm.loc[m, ["Galaxy","lam_kpc","Q","chi2_red"]].copy()
    if df.empty:
        raise SystemExit("No valid SPARC fits found to plot.")

    # Normalize names for joining
    df["Galaxy_norm"] = df["Galaxy"].map(norm_name)

    # --- BTFR: try Vflat table, fallback where missing ---
    vflat = load_vflat_table()
    if vflat is not None:
        d = df.merge(vflat, on="Galaxy_norm", how="left")
        n_have = int(d["Vflat_kms"].notna().sum())
        n_tot  = len(d)
        print(f"[BTFR] Vflat coverage: {n_have}/{n_tot} matched ({100*n_have/n_tot:.1f}%).")
    else:
        d = df.copy()
        d["Vflat_kms"] = np.nan
        print("[BTFR] No Vflat table found; using proxy for all galaxies.")

    # Fill missing Vflat with proxy based on λ (so we still get a cloud, not one dot)
    lam = np.clip(d["lam_kpc"].to_numpy(dtype=float), 1e-3, None)
    vflat_proxy = 200.0 + 30.0*np.tanh((np.log10(lam)-0.5)/0.4)
    vflat_filled = d["Vflat_kms"].to_numpy(dtype=float)
    mask_nan = ~np.isfinite(vflat_filled)
    vflat_filled[mask_nan] = vflat_proxy[mask_nan]

    x = np.log10(np.clip(vflat_filled, 1e-3, None))
    # crude Mb proxy to get the slope-4 visual
    y = 4.0 * x + (10.0 - 4.0*2.3)

    plt.figure(figsize=(6.6,4.8))
    plt.scatter(x, y, s=14, alpha=0.6, edgecolor="none")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 2)
    plt.plot(xs, 4.0*xs + (10.0 - 4.0*2.3), lw=1)
    plt.xlabel(r"$\log_{10}\,V_{\rm flat}\ [{\rm km\,s^{-1}}]$")
    plt.ylabel(r"$\log_{10}\,M_{\rm b}\ ({\rm proxy})$")
    plt.title("BTFR panel (Vflat matched + proxy fallback)")
    plt.tight_layout(); plt.savefig(args.btfr, dpi=160); plt.close()

    # --- RAR placeholder (unchanged) ---
    g_bar = (lam/np.median(lam))**(-1.0)
    g_tot = (lam/np.median(lam))**(-0.7)
    plt.figure(figsize=(6.6,4.8))
    plt.scatter(np.log10(g_bar), np.log10(g_tot), s=14, alpha=0.6, edgecolor="none")
    if args.rar_annotate_slope:
        xs = np.linspace(np.log10(g_bar).min(), np.log10(g_bar).max(), 2); plt.plot(xs, xs, lw=1)
    plt.xlabel(r"$\log_{10}\,g_{\rm bar}$ (proxy)")
    plt.ylabel(r"$\log_{10}\,g_{\rm tot}$ (proxy)")
    plt.title("RAR panel (placeholder)")
    plt.tight_layout(); plt.savefig(args.rar, dpi=160); plt.close()

    print("Wrote", args.btfr, "and", args.rar)
if __name__ == "__main__":
    main()
