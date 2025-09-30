import argparse, numpy as np, pandas as pd, numpy.linalg as LA
p = argparse.ArgumentParser()
p.add_argument("--kids", required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ell", type=float, default=1.3)
p.add_argument("--scale", type=float, default=1.0)
a = p.parse_args()
d = pd.read_csv(a.kids).to_numpy(float)
R, DS, E = d[:,0], d[:,1], d[:,2]
M = pd.read_csv(a.model).to_numpy(float)[:,1]
resid = M - DS
var = float(np.median(np.maximum(resid**2, E**2)))   # robust
lnR = np.log(np.clip(R,1e-12,None))
N = len(R); C = np.empty((N,N))
for i in range(N):
    for j in range(N):
        rho = np.exp(-abs(lnR[i]-lnR[j])/a.ell)
        C[i,j] = rho * var
try:
    LA.cholesky(C)
except LA.LinAlgError:
    eps = 1e-6*np.median(np.diag(C)); C += eps*np.eye(N)
C *= a.scale
pd.DataFrame(C).to_csv(a.out, header=False, index=False)
ev = LA.eigvalsh(C)
print(f"out={a.out} ell={a.ell} var~{var:.3g} eig(min,max)=({ev.min():.3g},{ev.max():.3g})")
