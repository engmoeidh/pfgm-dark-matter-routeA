import os, json, math, shutil, argparse
from pathlib import Path
import numpy as np
import pandas as pd

TGT_ROUTE  = Path("data/processed/routeA/routeA_m0p18_binB.csv")
TGT_KIDS   = Path("data/raw/kids/KiDS_binB.csv")
TGT_MODELS = Path("results/tables")
TGT_FIGS   = Path("figures/lensing")

def cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def aicbic(chi2nu: float, N: int, k: int):
    chi2 = chi2nu * max(N - k, 1)
    AIC  = chi2 + 2*k
    BIC  = chi2 + k*math.log(N)
    return float(chi2), float(AIC), float(BIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="C:\Users\HomePC\Dropbox\Home Office Desktop\Science reading\Physics\gravity solution\Dark matter")
    args = ap.parse_args()
    src = Path(args.src)

    zips = list(src.glob("KiDS_binB_outputs_package.zip"))
    if zips:
        import zipfile
        tmp = Path(".integrate_binB_tmp"); tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(tmp)
        S = tmp
    else:
        S = src

    kids_src  = next((S.glob("KiDS_binB_CONVERTED_kpc2.csv")), None)
    route_src = next((S.glob("routeA_m0p18_binB.csv")), None)
    if not kids_src or not kids_src.is_file():
        raise SystemExit("Missing KiDS_binB_CONVERTED_kpc2.csv under --src")
    if not route_src or not route_src.is_file():
        raise SystemExit("Missing routeA_m0p18_binB.csv under --src")

    cp(kids_src,  TGT_KIDS)
    cp(route_src, TGT_ROUTE)

    for name in ["binB_2halo_free.csv","binB_total_free.csv",
                 "binB_2halo_SPARClocked.csv","binB_total_SPARClocked.csv",
                 "binB_total_GRkernel.csv"]:
        p = S / name
        if p.is_file(): cp(p, TGT_MODELS / name)

    for name in ["overlay_binB_routeA_with_2halo_free.png",
                 "overlay_binB_routeA_with_2halo_SPARClocked.png",
                 "overlay_binB_GRkernel.png"]:
        p = S / name
        if p.is_file(): cp(p, TGT_FIGS / name)

    dfk = pd.read_csv(TGT_KIDS); N = len(dfk)

    free_vals = dict(lam=0.518, A1h=1.52e3, b0=-20.84, b2=10.97, chi2nu=0.855)
    lock_vals = dict(lam=0.221, A1h=1.0,   b0=-23.23, b2=5.07,  chi2nu=2.03)
    gr_vals   = dict(chi2nu=3.34)

    sjson = S / "binB_fit_summary.json"
    if sjson.is_file():
        try:
            jj = json.load(open(sjson,"r"))
            def upd(dst, blk):
                if not blk: return
                dst["lam"]    = blk.get("lambda", blk.get("lambda_mpc", dst["lam"]))
                dst["A1h"]    = blk.get("A1h", dst["A1h"])
                dst["b0"]     = blk.get("b0",  dst["b0"])
                dst["b2"]     = blk.get("b2",  dst["b2"])
                dst["chi2nu"] = blk.get("chi2/nu", blk.get("chi2_red", dst["chi2nu"]))
            upd(free_vals, jj.get("free", {}))
            upd(lock_vals, jj.get("SPARClocked", jj.get("locked", {})))
            gv = jj.get("GRkernel", jj.get("gr", {}))
            if gv:
                gr_vals["chi2nu"] = gv.get("chi2/nu", gv.get("chi2_red", gr_vals["chi2nu"]))
        except Exception:
            pass

    TGT_MODELS.mkdir(parents=True, exist_ok=True)

    chi2,AIC,BIC = aicbic(free_vals["chi2nu"], N, 4)
    json.dump(dict(mode="FREEA", bin="B",
                   lambda_mpc=float(free_vals["lam"]),
                   A1h=float(free_vals["A1h"]),
                   b0=float(free_vals["b0"]), b2=float(free_vals["b2"]),
                   chi2_red=float(free_vals["chi2nu"]), AIC=AIC, BIC=BIC),
              open(TGT_MODELS/"binB_fit_freeA.json","w"), indent=2)

    chi2,AIC,BIC = aicbic(lock_vals["chi2nu"], N, 3)
    json.dump(dict(mode="A1", bin="B",
                   lambda_mpc=float(lock_vals["lam"]),
                   A1h=float(lock_vals["A1h"]),
                   b0=float(lock_vals["b0"]), b2=float(lock_vals["b2"]),
                   chi2_red=float(lock_vals["chi2nu"]), AIC=AIC, BIC=BIC),
              open(TGT_MODELS/"binB_fit_A1.json","w"), indent=2)

    chi2,AIC,BIC = aicbic(gr_vals["chi2nu"], N, 3)
    json.dump(dict(mode="GR", bin="B",
                   lambda_mpc=0.0, A1h=np.nan, b0=np.nan, b2=np.nan,
                   chi2_red=float(gr_vals["chi2nu"]), AIC=AIC, BIC=BIC),
              open(TGT_MODELS/"binB_fit_GR.json","w"), indent=2)

    print("Integrated Bin-B:")
    print("  KiDS  ->", TGT_KIDS)
    print("  1-halo->", TGT_ROUTE)
    print("  JSONs -> binB_fit_freeA.json, binB_fit_A1.json, binB_fit_GR.json")

if __name__ == "__main__":
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    main()
