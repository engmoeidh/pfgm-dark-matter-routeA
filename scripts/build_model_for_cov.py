import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.special import jv

H=0.674; OM=0.315
RHO_M = 2.775e11*H*H*OM   # Msun/Mpc^3

def hankel_j2(k,Pgm,R):
    return np.array([np.trapz(k*Pgm*jv(2,k*r)/(2*np.pi), k) for r in R])

def load_align_1h(kids_csv, route_csv):
    d  = pd.read_csv(kids_csv).to_numpy(float)
    Rk = d[:,0]
    oneh = pd.read_csv(route_csv).to_numpy(float)
    R1, D1 = oneh[:,0], oneh[:,1]
    idx = np.argsort(R1); R1, D1 = R1[idx], D1[idx]
    DS1 = np.interp(np.log(np.clip(Rk,1e-6,None)),
                    np.log(np.clip(R1,1e-6,None)), D1)
    return Rk, DS1

def build_tot(R, DS1, k, P, A1h, b0, b2, lam, z):
    mu  = 1.0/(1.0 + (k*lam)**2) if lam>0 else np.ones_like(k)
    Pgm = (b0 + b2*k*k)*mu*P
    DS2 = RHO_M*hankel_j2(k,Pgm,R)/1e6
    DS2 *= (1.0+z)**2
    return A1h*DS1 + DS2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kids", required=True)        # data/raw/kids/KiDS_bin?.csv
    ap.add_argument("--route1h", required=True)     # data/processed/routeA/routeA_m0p18_bin?.csv (or *_on_KiDS.csv)
    ap.add_argument("--pnl", required=True)         # data/raw/Pnl_z0p286_HMcode.csv
    ap.add_argument("--fitjson", required=True)     # binA_fit_FREEA_J2_FULL.json or binB_fit_freeA.json
    ap.add_argument("--z", type=float, default=0.286)
    ap.add_argument("--out", required=True)         # results/tables/KiDS_bin?_model_for_cov.csv
    args=ap.parse_args()

    R, DS1 = load_align_1h(args.kids, args.route1h)
    kP = pd.read_csv(args.pnl).to_numpy(float); k, P = kP[:,0], kP[:,1]
    info = json.load(open(args.fitjson,"r"))

    # accept either our repo summary JSON or your earlier J2 JSON shape
    mode = (info.get("mode") or "").upper()
    A1h = info.get("A1h", 1.0 if mode=="A1" else info.get("A1h", 1.0))
    b0  = info.get("b0", 1.6)
    b2  = info.get("b2", 10.0)
    lam = info.get("lambda_mpc", info.get("lambda", 1.0))

    M = build_tot(R, DS1, k, P, A1h, b0, b2, lam, args.z)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"R":R, "DS_model":M}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}  N={len(R)}  A1h={A1h} b0={b0} b2={b2} lam={lam}")

if __name__=="__main__":
    main()
