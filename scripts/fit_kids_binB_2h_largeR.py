import argparse, numpy as np, pandas as pd, json, os
from scipy.optimize import least_squares
from scipy.special import jv

def hankel_j2(k,Pgm,R): return np.array([np.trapz(k*Pgm*jv(2,k*r)/(2*np.pi), k) for r in R])

ap = argparse.ArgumentParser()
ap.add_argument("--kids", required=True)
ap.add_argument("--pnl",  required=True)
ap.add_argument("--z", type=float, default=0.286)
ap.add_argument("--rmin", type=float, default=0.8)  # Mpc: cut to suppress 1h
ap.add_argument("--outjson", default="results/tables/binB_fit_2h_largeR.json")
a = ap.parse_args()

dat = pd.read_csv(a.kids).to_numpy(float)
R, D, E = dat[:,0], dat[:,1], dat[:,2]
m = R >= a.rmin
R, D, E = R[m], D[m], E[m]

tbl = pd.read_csv(a.pnl).to_numpy(float)
k, P = tbl[:,0], tbl[:,1]
h, Om = 0.674, 0.315
rho_m = 2.775e11*h*h*Om

def model(p):
    b0, b2, lam = p
    mu  = 1.0/(1.0 + (k*lam)**2)
    Pgm = (b0 + b2*k*k)*mu*P
    DS2 = rho_m*hankel_j2(k,Pgm,R)/1e6
    DS2 *= (1.0+a.z)**2
    return DS2

def resid(p): return (model(p)-D)/np.where(E>0,E,1.0)

p0  = np.array([1.6, 10.0, 1.0])
lo  = np.array([0.3,  0.0, 0.05])
hi  = np.array([5.0, 50.0, 10.0])

fit = least_squares(resid, p0, bounds=(lo,hi), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000)
b0, b2, lam = fit.x
chi2 = float(np.sum(resid(fit.x)**2))
nu   = len(R)-len(fit.x)
AIC  = chi2 + 2*len(fit.x); BIC = chi2 + len(fit.x)*np.log(len(R))

os.makedirs(os.path.dirname(a.outjson), exist_ok=True)
json.dump(dict(mode="B_2h_largeR", b0=b0, b2=b2, lambda_mpc=lam, rmin=a.rmin,
               chi2_red=chi2/nu, AIC=AIC, BIC=BIC), open(a.outjson,"w"), indent=2)
print("Fit large-R only:", {"b0":b0,"b2":b2,"lambda_mpc":lam,"chi2_red":chi2/nu,"rmin":a.rmin})
