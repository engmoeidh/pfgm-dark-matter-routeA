import argparse, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--kids", required=True, help="KiDS data CSV with columns R, DeltaSigma, sigma")
p.add_argument("--out",  required=True, help="Output PNG path")
p.add_argument("--label", default="", help="Legend title (e.g., cov-aware chi2/nu=...)")
p.add_argument("--total_free", default="", help="CSV with columns R, model(=DeltaSigma_tot) for free-A")
p.add_argument("--total_a1",   default="", help="CSV with columns R, model for SPARC-locked (A1)")
p.add_argument("--total_gr",   default="", help="CSV with columns R, model for GR-kernel control")
args = p.parse_args()

# --- Load KiDS ---
kids = pd.read_csv(args.kids).to_numpy(float)
R, DS = kids[:,0], kids[:,1]
E = kids[:,2] if kids.shape[1] >= 3 else np.zeros_like(DS)

plt.figure(figsize=(6,4.2))
plt.errorbar(R, DS, yerr=E, fmt='o', ms=4, lw=1, capsize=2, label="KiDS (data)")

# --- Helper to load and plot a model curve if provided ---
def plot_model(path, lbl):
    if not path: return False
    t = pd.read_csv(path).to_numpy(float)
    if t.shape[1] == 2:
        Rm, Mm = t[:,0], t[:,1]
    else:
        # try to detect columns by header names
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c.lower().startswith("r")]
        colm = [c for c in df.columns if "model" in c.lower() or "total" in c.lower()]
        if len(cols)==0 or len(colm)==0:
            raise RuntimeError(f"Unrecognized columns in {path}")
        Rm, Mm = df[cols[0]].to_numpy(float), df[colm[0]].to_numpy(float)
    idx = np.argsort(Rm)
    plt.plot(Rm[idx], Mm[idx], lw=2, label=lbl)
    return True

any_curve = False
any_curve |= plot_model(args.total_free, "PFGM total (free-A)")
any_curve |= plot_model(args.total_a1,   "PFGM total (A1 locked)")
any_curve |= plot_model(args.total_gr,   "GR-kernel total")

plt.xscale("log"); plt.xlabel(r"$R\ \mathrm{[Mpc]}$")
plt.ylabel(r"$\Delta\Sigma\ \mathrm{[M_\odot/kpc^2]}$")
if args.label:
    plt.legend(title=args.label, frameon=False)
else:
    plt.legend(frameon=False)
plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, dpi=180)
print("Wrote", args.out)
