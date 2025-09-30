import argparse, numpy as np, pandas as pd, numpy.linalg as LA
p = argparse.ArgumentParser()
p.add_argument("--cov", required=True)
p.add_argument("--n", type=int, default=15)
a = p.parse_args()
C = pd.read_csv(a.cov, header=None).to_numpy(float)
assert C.shape==(a.n,a.n), f"shape {C.shape} != ({a.n},{a.n})"
sym = float(np.abs(C-C.T).max())
ev  = LA.eigvalsh(C)
cond= float(ev.max()/ev.min()) if ev.min()>0 else float("inf")
print(f"shape={C.shape} max|C-C^T|={sym:.3e} eig(min,max)=({ev.min():.3g},{ev.max():.3g}) cond~{cond:.3g}")
