import argparse, os, json, re
from pathlib import Path
import numpy as np, pandas as pd

BIN_PATTERNS = [
    ("KiDS","A", r"\bbinA\b|\bbina\b|binA|bina"),
    ("KiDS","B", r"\bbinB\b|\bbinb\b|binB|binb"),
    ("KiDS","C", r"\bbinC\b|\bbinc\b|binC|binc"),
    ("SDSS","LOWZ",  r"\blowz\b|bin_lowz"),
    ("SDSS","CMASS", r"\bcmass\b|bin_cmass"),
]

def load_jsons(root: Path, bins_keep=None):
    rows = []
    for p in sorted(root.rglob("*.json")):
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
        # parse
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        # accept multiple key spellings
        lam   = d.get("lambda_mpc", d.get("lambda", np.nan))
        em    = d.get("lambda_err_minus", d.get("err_minus", np.nan))
        ep    = d.get("lambda_err_plus",  d.get("err_plus",  np.nan))
        chi2v = d.get("chi2_red", d.get("chi2nu", np.nan))
        edge  = int(d.get("edge", d.get("edge_flag", 0)))
        rows.append(dict(survey=survey, bin=binlab, lambda_mpc=lam,
                         lambda_err_minus=em, lambda_err_plus=ep,
                         edge=edge, chi2_red=chi2v, source=str(p)))
    cols = ["survey","bin","lambda_mpc","lambda_err_minus","lambda_err_plus","edge","chi2_red","source"]
    return pd.DataFrame(rows, columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids")
    ap.add_argument("--sdss")
    ap.add_argument("--bins", nargs="+")
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nboots", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out_json, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    bins_keep = set([b.upper() for b in args.bins]) if args.bins else None
    df = load_jsons(Path("results/tables"), bins_keep=bins_keep)

    if df.empty:
        # Emit explicit placeholder rows so downstream tables still exist
        print("WARN: no matching JSONs found; emitting placeholder rows.")
        df = pd.DataFrame(dict(
            survey=["SDSS","SDSS"], bin=["LOWZ","CMASS"],
            lambda_mpc=[np.nan, np.nan], lambda_err_minus=[np.nan, np.nan],
            lambda_err_plus=[np.nan, np.nan], edge=[np.nan, np.nan],
            chi2_red=[np.nan, np.nan], source=["",""]
        ))

    # keep the first row per bin with finite λ; else keep the first occurrence
    keep = []
    for (sv, bl), sub in df.groupby(["survey","bin"], sort=False):
        sub_f = sub.dropna(subset=["lambda_mpc"])
        s = (sub_f.iloc[0] if not sub_f.empty else sub.iloc[0]).to_dict()
        keep.append(s)
    out = pd.DataFrame(keep).sort_values(by=["survey","bin"]).reset_index(drop=True)

    out.to_csv(args.out_table, index=False)

    # copy jsons into normalized folder (best per bin), if any
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
