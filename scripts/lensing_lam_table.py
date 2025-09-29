import argparse, pandas as pd, numpy as np, os
ap = argparse.ArgumentParser()
ap.add_argument("--kids", required=True)
ap.add_argument("--sdss", required=True)
ap.add_argument("--out-tex", required=True)
ap.add_argument("--out-csv", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)

def norm(df, survey):
    df = df.copy()
    df["survey"] = survey
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    if "bin" not in df.columns:
        # infer bin from source file name if available
        if "source" in df.columns:
            df["bin"] = df["source"].str.extract(r'\b(bin[ABC]|LOWZ|CMASS)\b', expand=False).str.replace("bin","", case=False, regex=False).str.upper()
        else:
            df["bin"] = ""
    for k in ["lambda_mpc","lambda_err_minus","lambda_err_plus","edge","chi2_red"]:
        if k not in df.columns: df[k] = np.nan
    return df[["survey","bin","lambda_mpc","lambda_err_minus","lambda_err_plus","edge","chi2_red"]]

k = norm(pd.read_csv(a.kids), "KiDS")
s = norm(pd.read_csv(a.sdss), "SDSS")
all_ = pd.concat([k,s], ignore_index=True)
all_.to_csv(a.out_csv, index=False)

with open(a.out_tex,"w", encoding="utf-8") as f:
    f.write("\\begin{longtable}{llrrrrr}\n\\toprule\nSurvey & Bin & $\\lambda$ [Mpc] & $-\\sigma$ & $+\\sigma$ & edge & $\\chi^2/\\nu$\\\\\\midrule\n")
    for r in all_.itertuples():
        f.write(f"{r.survey} & {r.bin} & {getattr(r,'lambda_mpc',np.nan):.3g} & {getattr(r,'lambda_err_minus',np.nan):.2g} & {getattr(r,'lambda_err_plus',np.nan):.2g} & {int(getattr(r,'edge',0))} & {getattr(r,'chi2_red',np.nan):.2f}\\\\\n")
    f.write("\\bottomrule\n\\end{longtable}\n")
print("Wrote", a.out_csv, "and", a.out_tex)
