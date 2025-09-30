import h5py, numpy as np, pandas as pd, sys, os
src = "results/pm/PFGM_Pm_grid.hdf5"
dst = "data/raw/Pnl_z0p286_HMcode.csv"
os.makedirs("data/raw", exist_ok=True)
with h5py.File(src,"r") as h:
    k = h["k"][:]
    P = h["P"][0]   # first (and only) z-slice in the scaffold
pd.DataFrame({"k_1perMpc":k, "P_Mpc3":P}).to_csv(dst, index=False)
print("Wrote", dst, "nk=", len(k))
