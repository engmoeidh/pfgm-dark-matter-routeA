import json, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd

def find_bin(name:str, rec:dict) -> str:
    n = name.lower()
    if "bina" in n or re.search(r"\bbin[_-]?a\b", n): return "A"
    if "binb" in n or re.search(r"\bbin[_-]?b\b", n): return "B"
    if "binc" in n or re.search(r"\bbin[_-]?c\b", n): return "C"
    # fallback to record field
    b = str(rec.get("bin","")).strip().upper()
    return b if b in {"A","B","C"} else ""

def read_one(path:Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    name = path.stem
    b = find_bin(name, d)
    # extract lambda
    lam = d.get("lambda_mpc") or d.get("best_lambda") or d.get("lambda_best") or d.get("lambda") or d.get("lam")
    try: lam = float(lam)
    except: lam = np.nan
    # errors if present
    em = d.get("lambda_err_minus") or d.get("err_minus") or d.get("lam_err_minus") or d.get("lam_lo")
    ep = d.get("lambda_err_plus")  or d.get("err_plus")  or d.get("lam_err_plus")  or d.get("lam_hi")
    try: em = float(em)
    except: em = np.nan
    try: ep = float(ep)
    except: ep = np.nan
    # metrics
    chi2v = d.get("chi2_red") or d.get("chi2/nu")
    try: chi2v = float(chi2v)
    except: chi2v = np.nan
    AIC = d.get("AIC"); BIC = d.get("BIC")
    try: AIC = float(AIC)
    except: AIC = np.nan
    try: BIC = float(BIC)
    except: BIC = np.nan
    # mode label for pairing (A1/freeA/GR)
    mode = (d.get("mode") or "").upper()
    # heuristic: mark GR if lambda==0 or mode contains 'GR'
    is_gr = (np.isfinite(lam) and lam == 0.0) or ("GR" in mode)
    edge = 1 if (np.isfinite(lam) and lam >= 4.99) else 0
    return dict(bin=b, lambda_mpc=lam, lambda_err_minus=em, lambda_err_plus=ep,
                chi2_red=chi2v, AIC=AIC, BIC=BIC, is_gr=int(is_gr),
                edge=edge, source=str(path))

def main():
    roots = [Path("results/tables"), Path(".")]
    hits = []
    for root in roots:
        hits += list(root.rglob("*.json"))
    rows = []
    for p in sorted(hits):
        n = p.name.lower()
        # KiDS patterns only (bin A/B/C and not LOWZ/CMASS)
        if not (("bin" in n) and not ("lowz" in n or "cmass" in n)):
            continue
        rec = read_one(p)
        if rec and rec["bin"] in {"A","B","C"}:
            rows.append(rec)
    if not rows:
        # produce an empty but valid file so downstream doesn't crash
        out = pd.DataFrame(columns=["bin","lambda_mpc","lambda_err_minus","lambda_err_plus",
                                    "chi2_red","AIC","BIC","is_gr","edge","source"])
        out.to_csv("results/tables/KiDS_fit_summary_FINAL.csv", index=False)
        print("No KiDS JSONs found; wrote empty KiDS_fit_summary_FINAL.csv")
        return

    df = pd.DataFrame(rows)
    # pair within each bin: choose best non-GR row; compute ΔAIC/ΔBIC vs GR when available
    recs = []
    for b, sub in df.groupby("bin"):
        gr = sub[sub["is_gr"]==1]
        gr_AIC = float(gr["AIC"].iloc[0]) if len(gr) else np.nan
        gr_BIC = float(gr["BIC"].iloc[0]) if len(gr) else np.nan
        # pick best non-GR by lowest chi2_red (or whichever exists)
        non = sub[sub["is_gr"]==0]
        pick = (non.sort_values(["chi2_red","AIC","BIC"], na_position="last").iloc[0]
                if len(non) else sub.iloc[0])
        dAIC = (float(pick["AIC"])-gr_AIC) if np.isfinite(pick["AIC"]) and np.isfinite(gr_AIC) else np.nan
        dBIC = (float(pick["BIC"])-gr_BIC) if np.isfinite(pick["BIC"]) and np.isfinite(gr_BIC) else np.nan
        rec = pick.to_dict()
        rec["dAIC_vs_GR"] = dAIC
        rec["dBIC_vs_GR"] = dBIC
        recs.append(rec)

    out = pd.DataFrame(recs).sort_values("bin").reset_index(drop=True)
    out.to_csv("results/tables/KiDS_fit_summary_FINAL.csv", index=False)
    print("Wrote results/tables/KiDS_fit_summary_FINAL.csv with", len(out), "rows")
    print(out[["bin","lambda_mpc","chi2_red","dAIC_vs_GR","dBIC_vs_GR","edge","source"]])

if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)
    main()
