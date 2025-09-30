import argparse, os, sys
import pandas as pd
import numpy as np

def normalize(df):
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*keys):
        for k in keys:
            for kk in colmap:
                if kk == k.lower():
                    return colmap[kk]
        return None

    # explicitly include SDSS comoving names
    R  = pick("r","rp","r_mpc","radius","radius_mpc","r_com_mpc")
    DS = pick("deltasigma","delta_sigma","ds","sigmat",
              "delta_sigma_msun_kpc2","deltasigma_comoving_msun_per_pc2")
    DE = pick("deltasigma_err","delta_sigma_err","ds_err","sigmat_err",
              "sigma_deltasigma","deltasigma_comoving_err_msun_per_pc2")

    if R is None or DS is None or DE is None:
        raise ValueError(f"Missing needed columns in {list(df.columns)}")

    out = df.rename(columns={R:"R", DS:"DeltaSigma", DE:"DeltaSigma_err"})[["R","DeltaSigma","DeltaSigma_err"]]
    out = out.replace([np.inf,-np.inf], np.nan).dropna()
    out = out[out["R"]>0].sort_values("R").reset_index(drop=True)
    out = out[out["DeltaSigma_err"]>0]
    if not (out["R"].values[1:] > out["R"].values[:-1]).all():
        raise ValueError("R must be strictly increasing after clean.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not os.path.isfile(args.input):
        print("Missing", args.input); sys.exit(2)
    df = pd.read_csv(args.input)
    out = normalize(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")

if __name__ == "__main__":
    main()
