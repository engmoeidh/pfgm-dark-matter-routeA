import json, glob, numpy as np, pandas as pd, os
os.makedirs("results/tables", exist_ok=True)

rows=[]
for p in glob.glob("results/tables/bin_*LOWZ*json")+glob.glob("results/tables/bin_*CMASS*json"):
    d=json.load(open(p,"r"))
    b = "LOWZ" if "LOWZ" in p.upper() else ("CMASS" if "CMASS" in p.upper() else "")
    lam = d.get("lambda_mpc"); 
    if lam is None: lam = d.get("best_lambda") or d.get("lambda_best") or d.get("lambda") or d.get("lam")
    try: lam = float(lam)
    except: lam = np.nan
    chi2v = d.get("chi2_red")
    if chi2v is None: chi2v = d.get("chi2/nu")
    try: chi2v = float(chi2v)
    except: chi2v = np.nan
    edge = 1 if (np.isfinite(lam) and lam>=4.99) else 0
    rows.append(dict(survey="SDSS", bin=b, lambda_mpc=lam,
                     lambda_err_minus=np.nan, lambda_err_plus=np.nan,
                     edge=edge, chi2_red=chi2v, source=p))

out = pd.DataFrame(rows).sort_values(["bin"])
out.to_csv("results/tables/SDSS_fit_summary_FINAL.csv", index=False)
print("Wrote results/tables/SDSS_fit_summary_FINAL.csv with", len(out), "rows")
