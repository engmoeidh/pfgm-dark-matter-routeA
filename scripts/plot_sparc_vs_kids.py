import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SPARC_CSV = "results/tables/sparc_lambda_summary.csv"
JSON_A = "results/tables/bin_A_fit_freeA.json"
JSON_B = "results/tables/bin_B_fit_freeA.json"

def read_kids(path):
    try:
        return float(json.load(open(path,"r"))["lam"]) if os.path.isfile(path) else None
    except Exception:
        return None

def main():
    if not os.path.isfile(SPARC_CSV):
        print("Missing", SPARC_CSV); return
    df = pd.read_csv(SPARC_CSV)
    sel = df[(df["lam_kpc"].notna()) & (df["status"].str.startswith(("OK","EDGE")))]
    if sel.empty:
        print("No SPARC rows with finite lambda."); return
    lam_mpc = sel["lam_kpc"].to_numpy(float) / 1000.0

    lamA, lamB = read_kids(JSON_A), read_kids(JSON_B)

    os.makedirs("figures/sparc", exist_ok=True)
    plt.figure(figsize=(6.8,4.2))
    n,b,_ = plt.hist(lam_mpc, bins=24, alpha=0.85, edgecolor="k", label="SPARC (λ/1000)")
    ymin, ymax = 0, max(n)*1.15 if n.size else 1.0

    def band(x, color, label):
        if x is None: return
        w = 0.1*x
        plt.fill_between([x-w, x+w], ymin, ymax, color=color, alpha=0.15, label=label)
        plt.axvline(x, color=color, lw=2)

    band(lamA, "C1", "KiDS A (free-A)")
    band(lamB, "C2", "KiDS B (free-A)")

    plt.ylim(ymin, ymax)
    plt.xlabel(r"$\lambda$ [Mpc]")
    plt.ylabel("Number of galaxies")
    plt.title(r"SPARC vs KiDS: $\lambda$ comparison")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = "figures/sparc/sparc_vs_kids_lambda.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("Wrote", out, f"(N={len(sel)})")

if __name__ == "__main__":
    main()
