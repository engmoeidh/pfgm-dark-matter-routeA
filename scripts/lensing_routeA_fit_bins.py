import argparse, os, json, re
from pathlib import Path
import numpy as np, pandas as pd

BIN_PATTERNS = [
    ("KiDS","A", r"\bbinA\b|\bbina\b|binA|bina"),
    ("KiDS","B", r"\bbinB\b|\bbinb\b|binB|binb"),
    ("KiDS","C", r"\bbinC\b|\bbinc\b|binC|binc"),
    ("SDSS","LOWZ",  r"\blowz\b"),
    ("SDSS","CMASS", r"\bcmass\b"),
]

def load_existing_jsons(root: Path, bins_keep=None):
    rows = []
    root = Path(root)
    for p in sorted(root.glob("*.json")):
        name = p.stem.lower()
        survey, binlab = None, None
        for (sv, bl, pat) in BIN_PATTERNS:
            if re.search(pat, name, flags=re.I):
                survey, binlab = sv, bl
                break
        if binlab is None:
            continue
        if bins_keep and binlab.upper() not in bins_keep:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        lam   = d.get("lambda_mpc", d.get("lambda", np.nan))
        em    = d.get("lambda_err_minus", d.get("err_minus", np.nan))
        ep    = d.get("lambda_err_plus",  d.get("err_plus",  np.nan))
        chi2v = d.get("chi2_red", np.nan)
        edge  = int(d.get("edge", d.get("edge_flag", 0)))
        rows.append(dict(survey=survey, bin=binlab, lambda_mpc=lam,
                         lambda_err_minus=em, lambda_err_plus=ep,
                         edge=edge, chi2_red=chi2v, source=str(p)))
    cols = ["survey","bin","lambda_mpc","lambda_err_minus","lambda_err_plus","edge","chi2_red","source"]
    return pd.DataFrame(rows, columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids", help="(unused in collector mode)")
    ap.add_argument("--sdss", help="(unused in collector mode)")
    ap.add_argument("--bins", nargs="+", help="A B C or LOWZ CMASS")
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nboots", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out_json, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    bins_keep = set([b.upper() for b in args.bins]) if args.bins else None
    df = load_existing_jsons(Path("results/tables"), bins_keep=bins_keep)

    if df.empty:
        raise SystemExit(
            "No per-bin JSONs found in results/tables/*.json matching {binA,B,C,LOWZ,CMASS}.\n"
            "Drop files like: binA_fit_A1.json, binA_fit_freeA.json, LOWZ_fit_*.json, CMASS_fit_*.json."
        )

    # If multiple JSONs per bin exist, keep the first with finite λ; else mean combine
    keep = []
    for (sv, bl), sub in df.groupby(["survey","bin"], sort=False):
        sub = sub.dropna(subset=["lambda_mpc"])
        if sub.empty:
            keep.append(dict(survey=sv, bin=bl, lambda_mpc=np.nan,
                             lambda_err_minus=np.nan, lambda_err_plus=np.nan,
                             edge=np.nan, chi2_red=np.nan, source=""))
        else:
            s = sub.iloc[0].to_dict()
            keep.append(s)
    out = pd.DataFrame(keep)
    out = out.sort_values(by=["survey","bin"]).reset_index(drop=True)
    out.to_csv(args.out_table, index=False)

    # Copy JSONs (best per bin) into normalized folder
    from shutil import copy2
    for r in out.itertuples():
        if r.source:
            dst = Path(args.out_json)/Path(r.source).name
            if not dst.exists():
                copy2(r.source, dst)

    print(f"Wrote {args.out_table} with {len(out)} rows")
    print(f"Collected JSONs in {args.out_json}")
if __name__ == "__main__":
    main()
