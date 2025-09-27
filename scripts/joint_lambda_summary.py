import json, glob, os
import pandas as pd

def grab(js):
    d = json.load(open(js,"r"))
    return {
        "file": js,
        "bin":  d.get("bin"),
        "mode": d.get("mode"),
        "A1h":  d.get("A1h"),
        "b0":   d.get("b0"),
        "b2":   d.get("b2"),
        "lam":  d.get("lam"),
        "chi2": d.get("chi2"),
        "nu":   d.get("nu"),
        "chi2/nu": d.get("chi2/nu"),
        "AIC":  d.get("AIC"),
        "BIC":  d.get("BIC"),
        "ΔAIC_vs_GR": d.get("ΔAIC_vs_GR"),
        "ΔBIC_vs_GR": d.get("ΔBIC_vs_GR"),
    }

def main():
    files = sorted(glob.glob("results/tables/bin_*_fit_*.json"))
    rows  = [grab(f) for f in files]
    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    out = "results/tables/lambda_consistency_A_B.csv"
    df.to_csv(out, index=False)
    print("Wrote", out)
    try:
        print(df.pivot_table(index="bin", columns="mode", values="lam"))
    except Exception as e:
        print("Pivot failed:", e)

if __name__ == "__main__":
    main()
