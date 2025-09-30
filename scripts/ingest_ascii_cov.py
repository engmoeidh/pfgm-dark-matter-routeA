import argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--inpath", required=True)
p.add_argument("--outcsv", required=True)
a = p.parse_args()
C = np.loadtxt(a.inpath)
assert C.ndim==2 and C.shape[0]==C.shape[1], "Matrix must be square."
C = 0.5*(C + C.T)  # symmetrize tiny drift
pd.DataFrame(C).to_csv(a.outcsv, header=False, index=False)
print("Wrote", a.outcsv, "shape", C.shape)
