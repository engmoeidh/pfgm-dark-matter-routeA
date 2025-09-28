import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt

SPARC = "results/tables/sparc_lambda_summary.csv"
BINS  = [("A","C1"), ("B","C2"), ("LOWZ","C3"), ("CMASS","C4")]
def lam_from_json(j): return float(json.load(open(j))["lam"])

def main():
    if not os.path.isfile(SPARC): 
        print("Missing", SPARC); return
    df = pd.read_csv(SPARC)
    ok = df[(df["lam_kpc"].notna()) & (df["status"].str.startswith(("OK","EDGE")))]
    if ok.empty:
        print("No SPARC λ to plot."); return
    lam_mpc = ok["lam_kpc"].to_numpy(float)/1000.0

    # collect bin λ (free-A) if JSON exists
    bands = []
    for b, col in BINS:
        j = f"results/tables/bin_{b}_fit_freeA.json"
        bands.append((b, col, lam_from_json(j) if os.path.isfile(j) else None))

    os.makedirs("figures/sparc", exist_ok=True)
    plt.figure(figsize=(7.2,4.2))
    n,b,_ = plt.hist(lam_mpc, bins=30, alpha=0.85, edgecolor="k", label="SPARC (λ/1000)")
    ymin, ymax = 0, max(n)*1.15 if n.size else 1.0

    for b, col, x in bands:
        if x is None: continue
        w = 0.1*x
        plt.fill_between([x-w, x+w], ymin, ymax, color=col, alpha=0.15, label=f"{b} (free-A)")
        plt.axvline(x, color=col, lw=2)

    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\lambda$ [Mpc]"); plt.ylabel("Number of galaxies")
    plt.title(r"SPARC vs KiDS + SDSS: $\lambda$ comparison")
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    out = "figures/sparc/sparc_vs_kids_sdss_lambda.png"
    plt.savefig(out, dpi=160); plt.close()
    print("Wrote", out)

if __name__ == "__main__":
    main()
