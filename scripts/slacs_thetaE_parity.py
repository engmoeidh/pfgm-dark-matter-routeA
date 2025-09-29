import argparse, io, os, re, zipfile
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def norm_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s.upper()

def read_csv_forgiving(p: Path):
    for kw in (dict(sep=None, engine="python"),
               dict(sep=",", engine="python"),
               dict(sep=r"\s+", engine="python"),
               dict()):
        try:
            return pd.read_csv(p, **kw)
        except Exception:
            pass
    return None

def read_compare_table(path_csv: Path) -> pd.DataFrame:
    df = read_csv_forgiving(path_csv)
    if df is None or df.empty:
        raise SystemExit(f"Cannot read comparison file: {path_csv}")
    df.columns = [str(c).strip() for c in df.columns]
    # lens name column
    cand = [c for c in df.columns if re.search(r'(lens|name|id)', c, re.I)]
    name_col = cand[0] if cand else df.columns[0]
    df = df.rename(columns={name_col: "lens"})
    df["lens_norm"] = df["lens"].map(norm_name)
    # theta_E & errors (many possible aliases)
    def pick(prefixes, pat):
        for c in df.columns:
            if any(c.lower().startswith(pfx) for pfx in prefixes) or re.search(pat, c, re.I):
                return c
        return None
    th_gr   = pick(["thetae_gr","thetaegr","ein_gr","gr_thetae","thetae_ref"], r"(theta|ein).*gr")
    th_pf   = pick(["thetae_pfgm","thetaepfgm","ein_pfgm","pfgm_thetae"], r"(theta|ein).*(pfgm|model)")
    e_gr    = pick(["sigma_gr","err_gr","unc_gr","thetae_gr_err"], r"(err|sig|unc).*(gr)")
    e_pf    = pick(["sigma_pfgm","err_pfgm","unc_pfgm","thetae_pfgm_err"], r"(err|sig|unc).*(pfgm|model)")
    if th_gr:   df = df.rename(columns={th_gr: "thetaE_GR"})
    if th_pf:   df = df.rename(columns={th_pf: "thetaE_PFGM"})
    if e_gr:    df = df.rename(columns={e_gr: "sigma_GR"})
    if e_pf:    df = df.rename(columns={e_pf: "sigma_PFGM"})
    for k in ("thetaE_GR","thetaE_PFGM","sigma_GR","sigma_PFGM"):
        if k not in df.columns: df[k] = np.nan
    return df[["lens","lens_norm","thetaE_GR","thetaE_PFGM","sigma_GR","sigma_PFGM"]].copy()

def read_parity_from_zip(path_zip: Path) -> pd.DataFrame:
    if not path_zip.is_file():
        return pd.DataFrame(columns=["lens_norm","parity"])
    with zipfile.ZipFile(path_zip, "r") as zf:
        cand = [n for n in zf.namelist() if re.search(r"parity.*\.csv$", n, re.I)]
        if not cand:
            cand = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not cand:
                return pd.DataFrame(columns=["lens_norm","parity"])
        with zf.open(cand[0]) as fh:
            raw = fh.read()
        df = None
        for sep in (None, ",", r"\s+"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
                break
            except Exception:
                pass
        if df is None or df.empty:
            return pd.DataFrame(columns=["lens_norm","parity"])
        df.columns = [str(c).strip() for c in df.columns]
        name_col = [c for c in df.columns if re.search(r'(lens|name|id)', c, re.I)]
        par_col  = [c for c in df.columns if re.search(r'parit', c, re.I)]
        if not name_col or not par_col:
            return pd.DataFrame(columns=["lens_norm","parity"])
        out = df[[name_col[0], par_col[0]]].copy()
        out.columns = ["lens","parity"]
        out["lens_norm"] = out["lens"].map(norm_name)
        return out[["lens_norm","parity"]].drop_duplicates()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=False, help="SLACS baseline bundle zip (optional parity)")
    ap.add_argument("--compare", required=False, help="CSV with GR vs PFGM theta_E and uncertainties")
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--blind", type=float, default=0.0, help="fractional blind jitter on y only (plot)")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    os.makedirs(Path(args.outcsv).parent, exist_ok=True)
    os.makedirs(Path(args.fig).parent, exist_ok=True)

    # Auto-discover compare CSV if not provided
    if args.compare and Path(args.compare).is_file():
        comp_path = Path(args.compare)
    else:
        hits = [p for p in glob("data/raw/**/*.csv", recursive=True)
                if ("theta" in p.lower() and "e" in p.lower()) or ("ein" in p.lower())]
        if not hits:
            raise SystemExit("No comparison CSV found. Provide --compare pointing to GR vs PFGM θ_E file.")
        comp_path = Path(hits[0])
        print(f"[auto] Using compare file: {comp_path}")

    comp = read_compare_table(comp_path)
    par  = read_parity_from_zip(Path(args.bundle)) if args.bundle else __import__("pandas").DataFrame(columns=["lens_norm","parity"])
    df   = comp.merge(par, on="lens_norm", how="left")

    th_gr = pd.to_numeric(df["thetaE_GR"], errors="coerce")
    th_pf = pd.to_numeric(df["thetaE_PFGM"], errors="coerce")
    sg_gr = pd.to_numeric(df["sigma_GR"], errors="coerce")
    sg_pf = pd.to_numeric(df["sigma_PFGM"], errors="coerce")

    dth     = th_pf - th_gr
    sig_cmb = np.sqrt(np.clip(sg_gr,0, np.inf)**2 + np.clip(sg_pf,0, np.inf)**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigN = dth / np.where(sig_cmb>0, sig_cmb, np.nan)

    out = pd.DataFrame(dict(
        lens=df["lens"],
        thetaE_GR=th_gr,
        thetaE_PFGM=th_pf,
        sigma_GR=sg_gr,
        sigma_PFGM=sg_pf,
        thetaE_shift=dth,
        thetaE_shift_sigma=sigN,
        parity=df.get("parity", np.nan),
    ))
    out.to_csv(args.outcsv, index=False)
    print(f"Wrote {args.outcsv}  (N={len(out)})")
    print("⟨ΔθE/σ⟩ =", np.nanmean(out["thetaE_shift_sigma"]))

    xs = th_gr.to_numpy(dtype=float)
    ys = th_pf.to_numpy(dtype=float)
    if args.blind and args.blind>0:
        rng = np.random.default_rng(args.seed)
        ys_plot = ys * (1.0 + args.blind * rng.normal(size=ys.shape))
    else:
        ys_plot = ys

    plt.figure(figsize=(5.2,5.0))
    plt.scatter(xs, ys_plot, s=16, alpha=0.7, edgecolor="none")
    finite = np.isfinite(xs) & np.isfinite(ys_plot)
    if finite.any():
        lo = np.nanmin(np.r_[xs[finite], ys_plot[finite]])
        hi = np.nanmax(np.r_[xs[finite], ys_plot[finite]])
        grid = np.linspace(lo, hi, 2)
        plt.plot(grid, grid, lw=1, alpha=0.8)
    plt.xlabel(r"$\theta_E^{\rm GR}$")
    plt.ylabel(r"$\theta_E^{\rm PFGM}$ (blinded)" if args.blind>0 else r"$\theta_E^{\rm PFGM}$")
    plt.title("SLACS Einstein radius comparison")
    plt.tight_layout(); plt.savefig(args.fig, dpi=160); plt.close()
    print(f"Wrote {args.fig}")

if __name__ == "__main__":
    main()
