import argparse, os, numpy as np, pandas as pd, h5py
ap=argparse.ArgumentParser()
ap.add_argument("--out"); ap.add_argument("--validate"); ap.add_argument("--out-summary")
a=ap.parse_args()
if a.out:
  # try repo Pnl, else your Desktop turnkey path
  try: df=pd.read_csv("data/raw/Pnl_z0p286_HMcode.csv")
  except: df=pd.read_csv(r"C:/Users/HomePC/Desktop/Data/kids_ggl_turnkey/PFGM_NL/Pnl_z0p286_HMcode.csv")
  k=df.iloc[:,0].to_numpy(float); P=df.iloc[:,1].to_numpy(float)
  os.makedirs(os.path.dirname(a.out),exist_ok=True)
  with h5py.File(a.out,"w") as h:
    h.create_dataset("z",data=np.array([0.286]))
    h.create_dataset("k",data=k); h.create_dataset("P",data=P[None,:])
  print("Wrote",a.out,"nz=1 nk=",len(k))
elif a.validate and a.out_summary:
  with h5py.File(a.validate,"r") as h:
    z=h["z"][:]; k=h["k"][:]; P=h["P"][:]
  os.makedirs(os.path.dirname(a.out_summary),exist_ok=True)
  open(a.out_summary,"w").write(f"nz={len(z)} nk={len(k)} sigma8_eff~{np.sqrt((P/k**3).mean()):.3f}\n")
  print("Wrote",a.out_summary)
