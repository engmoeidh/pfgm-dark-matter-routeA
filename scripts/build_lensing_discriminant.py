import json, glob, os, re
import numpy as np, pandas as pd

def load_rows(patterns):
    rows=[]
    for pat in patterns:
        for p in glob.glob(pat):
            name = os.path.basename(p).lower()
            d = json.load(open(p,"r"))
            # identify survey/bin
            if "lowz" in name: survey, b = "SDSS","LOWZ"
            elif "cmass" in name: survey, b = "SDSS","CMASS"
            elif re.search(r"\bbin[_-]?a\b", name) or "bina" in name: survey, b = "KiDS","A"
            elif re.search(r"\bbin[_-]?b\b", name) or "binb" in name: survey, b = "KiDS","B"
            elif re.search(r"\bbin[_-]?c\b", name) or "binc" in name: survey, b = "KiDS","C"
            else: continue
            # read stats
            lam = d.get("lambda_mpc") or d.get("best_lambda") or d.get("lambda_best") or d.get("lambda") or d.get("lam")
            AIC = d.get("AIC"); BIC = d.get("BIC"); chi2nu = d.get("chi2_red") or d.get("chi2/nu")
            mode = (d.get("mode") or "").upper()
            is_gr = (mode.find("GR")>=0) or (lam==0 or lam=="0")
            try: lam = float(lam)
            except: lam = np.nan
            try: AIC = float(AIC)
            except: AIC = np.nan
            try: BIC = float(BIC)
            except: BIC = np.nan
            try: chi2nu = float(chi2nu)
            except: chi2nu = np.nan
            rows.append(dict(survey=survey, bin=b, is_gr=int(is_gr), lam=lam, AIC=AIC, BIC=BIC, chi2nu=chi2nu, source=p))
    return pd.DataFrame(rows)

def best_non_gr(sub: pd.DataFrame):
    non = sub[sub["is_gr"]==0]
    if len(non)==0: return None
    # rank by AIC, then BIC, then chi2nu
    return non.sort_values(["AIC","BIC","chi2nu"], na_position="last").iloc[0]

def main():
    df = load_rows(["results/tables/*.json", "results/tables/**/*json"])
    out_rows=[]
    for (sv,b), sub in df.groupby(["survey","bin"]):
        gr = sub[sub["is_gr"]==1]
        pick = best_non_gr(sub)
        if pick is None:
            continue
        AIC_gr = float(gr["AIC"].iloc[0]) if ("AIC" in gr and len(gr) and np.isfinite(gr["AIC"].iloc[0])) else np.nan
        BIC_gr = float(gr["BIC"].iloc[0]) if ("BIC" in gr and len(gr) and np.isfinite(gr["BIC"].iloc[0])) else np.nan
        dAIC   = float(pick["AIC"])-AIC_gr if np.isfinite(pick["AIC"]) and np.isfinite(AIC_gr) else np.nan
        dBIC   = float(pick["BIC"])-BIC_gr if np.isfinite(pick["BIC"]) and np.isfinite(BIC_gr) else np.nan
        out_rows.append(dict(
            survey=sv, bin=b, lam_best=float(pick["lam"]),
            chi2nu_best=float(pick["chi2nu"]), dAIC_vs_GR=dAIC, dBIC_vs_GR=dBIC,
            best_source=pick["source"]
        ))
    out = pd.DataFrame(out_rows).sort_values(["survey","bin"]).reset_index(drop=True)
    os.makedirs("results/tables", exist_ok=True)
    out.to_csv("results/tables/lensing_discriminant.csv", index=False)
    print(out)
if __name__ == "__main__":
    main()
