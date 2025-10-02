import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.special import jv

H=0.674; OM=0.315
RHO_M = 2.775e11*H*H*OM   # Msun/Mpc^3

def hankel_j2(k,Pgm,R):
    # ? dk [k Pgm J2(kR)/(2p)]  ->  Mpc
    return np.array([np.trapz(k*Pgm*jv(2,k*r)/(2*np.pi), k) for r in R])

def load_align_1h(kids_csv, route_csv, ds1_unit="Msun/kpc^2"):
    """
    Returns:
      Rk [Mpc, comoving], DS1 [Msun/kpc^2, comoving-area convention]
    """
    d  = pd.read_csv(kids_csv).to_numpy(float)
    Rk = d[:,0]  # KiDS radii (Mpc, comoving)
    oneh = pd.read_csv(route_csv).to_numpy(float)
    R1, D1 = oneh[:,0], oneh[:,1]

    # Convert DS1 to Msun/kpc^2 if needed
    u = ds1_unit.lower().replace(" ","")
    if u in ("msun/mpc^2","msun/mpc2"):
        D1 = D1/1.0e6     # Mpc^-2 -> kpc^-2
    elif u in ("msun/pc^2","msun/pc2"):
        D1 = D1*1.0e6     #  pc^-2 -> kpc^-2
    elif u not in ("msun/kpc^2","msun/kpc2"):
        raise ValueError(f"Unrecognized ds1_unit={ds1_unit}")

    # Interpolate (log-R safe)
    idx = np.argsort(R1); R1, D1 = R1[idx], D1[idx]
    DS1 = np.interp(np.log(np.clip(Rk,1e-6,None)),
                    np.log(np.clip(R1,1e-6,None)), D1)
    return Rk, DS1

def build_tot(R, DS1, k, P, A1h, b0, b2, lam, z):
    # PFGM lensing response in k-space
    mu  = 1.0/(1.0 + (k*lam)**2) if lam>0 else np.ones_like(k)
    Pgm = (b0 + b2*k*k)*mu*P      # Mpc^3

    # 2-halo: ?_m [Msun/Mpc^3]  hankel_j2 [Mpc] -> Msun/Mpc^2; /1e6  Msun/kpc^2
    DS2 = RHO_M*hankel_j2(k,Pgm,R)/1e6
    # exact J2 + (1+z)^2 (comoving-area convention)
    DS2 *= (1.0+z)**2

    # total (both in Msun/kpc^2, comoving-area convention)
    return A1h*DS1 + DS2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kids",     required=True, help="KiDS CSV used only for R-grid [Mpc, comoving]")
    ap.add_argument("--route1h",  required=True, help="Route-A 1h CSV [R, DS_1h] (DS_1h unit specified by --ds1_unit)")
    ap.add_argument("--pnl",      required=True, help="HMcode P_nl CSV [k (1/Mpc), P (Mpc^3)]")
    ap.add_argument("--fitjson",  required=True, help="fit JSON with A1h, b0, b2, lambda_mpc")
    ap.add_argument("--z",        type=float, default=0.286)
    ap.add_argument("--ds1_unit", default="Msun/kpc^2", help="Msun/kpc^2 (default), Msun/Mpc^2, or Msun/pc^2")
    ap.add_argument("--out",      required=True, help="Output CSV: R [Mpc], DS_model [Msun/kpc^2] (comoving)")
    args=ap.parse_args()

    # R-grid & 1h (converted to Msun/kpc^2)
    R, DS1 = load_align_1h(args.kids, args.route1h, ds1_unit=args.ds1_unit)

    # HMcode P_nl
    kP = pd.read_csv(args.pnl).to_numpy(float); k, P = kP[:,0], kP[:,1]

    # Fit params
    info = json.load(open(args.fitjson,"r"))
    mode = (info.get("mode") or "").upper()
    A1h  = info.get("A1h", 1.0 if mode=="A1" else info.get("A1h", 1.0))
    b0   = info.get("b0", 1.6)
    b2   = info.get("b2", 10.0)
    lam  = info.get("lambda_mpc", info.get("lambda", 1.0))

    M = build_tot(R, DS1, k, P, A1h, b0, b2, lam, args.z)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"R":R, "DS_model":M}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}  N={len(R)}  A1h={A1h} b0={b0} b2={b2} lam={lam}  ds1_unit={args.ds1_unit}")

if __name__=="__main__":
    main()
