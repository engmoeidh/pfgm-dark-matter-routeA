#!/usr/bin/env python
import argparse, numpy as np, pandas as pd

def comoving_chi(z,h,Om,Ol):
    c=299792.458; H0=100*h
    zz=np.linspace(0,z,4001); Ez=np.sqrt(Om*(1+zz)**3+Ol)
    return (c/H0)*np.trapz(1/Ez,zz)
def DA(z,h,Om,Ol): return 0 if z<=0 else comoving_chi(z,h,Om,Ol)/(1+z)
def sigma_c_inv(zl,zs,h,Om,Ol):
    if zs<=zl: return 0.0
    G=4.30091e-6; c=299792.458
    Dl=DA(zl,h,Om,Ol); Ds=DA(zs,h,Om,Ol)
    Dls=(comoving_chi(zs,h,Om,Ol)-comoving_chi(zl,h,Om,Ol))/(1+zs)
    Dl,Ds,Dls=1e3*Dl,1e3*Ds,1e3*Dls
    return (4*np.pi*G/c**2)*(Dl*Dls/Ds)
def norm_nz(z,n):
    I=np.trapz(n,z); 
    if I<=0: raise ValueError("n(z) integral is zero.")
    return z, n/I
def read_lens_txt(p):
    a=np.loadtxt(p); 
    if a.ndim!=2 or a.shape[1]<2: raise RuntimeError(f"Bad lens txt: {p}")
    return norm_nz(a[:,0], a[:,1])
def read_source_txt(p):
    a=np.loadtxt(p); 
    if a.ndim!=2 or a.shape[1]<2: raise RuntimeError(f"Bad source txt: {p}")
    return norm_nz(a[:,0], a[:,1])
def avg_scinv(zl,nl,zs,ns,h,Om,Ol):
    ZL,ZS=np.meshgrid(zl,zs,indexing="ij")
    NL,NS=np.meshgrid(nl,ns,indexing="ij")
    v=np.vectorize(lambda zL,zS: sigma_c_inv(zL,zS,h,Om,Ol))
    S=v(ZL,ZS); W=NL*NS
    den=np.trapz(np.trapz(W,zs,axis=1), zl, axis=0)
    num=np.trapz(np.trapz(W*S,zs,axis=1), zl, axis=0)
    if den<=0: raise ValueError("Zero denom in <Sigma_c^{-1}>.")
    return num/den
def build(bin_model, lens_txt, source_txt, zl, h, Om, Ol, out_model, out_sc):
    M=pd.read_csv(bin_model)
    Rcom=M.iloc[:,0].to_numpy(float)              # Mpc (comoving)
    DScom=M.iloc[:,1].to_numpy(float)             # Msun/kpc^2 (comoving convention)
    Dl=DA(zl,h,Om,Ol); Dl_kpc=1e3*Dl
    a_l=1/(1+zl)
    zL,nL=read_lens_txt(lens_txt)
    zS,nS=read_source_txt(source_txt)
    SC=avg_scinv(zL,nL,zS,nS,h,Om,Ol)            # kpc^2/Msun
    Rphys=Rcom/(1+zl)                             # Mpc
    theta_rad=(1e3*Rphys)/Dl_kpc
    theta_arcmin=theta_rad*(180/np.pi)*60
    DSphys=DScom/(a_l*a_l)
    gt=DSphys*SC                                  # dimensionless
    pd.DataFrame({"theta_arcmin":theta_arcmin,"gammat_model":gt}).to_csv(out_model,index=False)
    pd.DataFrame({"Sigma_c_inv_kpc2_perMsun":[SC]}).to_csv(out_sc,index=False)
    return theta_arcmin.size, SC
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--binA_model", required=True)
    ap.add_argument("--binB_model", required=True)
    ap.add_argument("--lens_nz_A", required=True)
    ap.add_argument("--lens_nz_B", required=True)
    ap.add_argument("--source_nz_path", required=True) # 2-col TXT (z, n(z))
    ap.add_argument("--zl", type=float, required=True)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--Om", type=float, default=0.315)
    ap.add_argument("--Ol", type=float, default=0.685)
    ap.add_argument("--outA", default="results/tables/KiDS_binA_gammat_model.csv")
    ap.add_argument("--outB", default="results/tables/KiDS_binB_gammat_model.csv")
    ap.add_argument("--outSigmaA", default="results/tables/KiDS_SigmaCinv_binA.csv")
    ap.add_argument("--outSigmaB", default="results/tables/KiDS_SigmaCinv_binB.csv")
    a=ap.parse_args()
    nA,SC_A=build(a.binA_model,a.lens_nz_A,a.source_nz_path,a.zl,a.h,a.Om,a.Ol,a.outA,a.outSigmaA)
    nB,SC_B=build(a.binB_model,a.lens_nz_B,a.source_nz_path,a.zl,a.h,a.Om,a.Ol,a.outB,a.outSigmaB)
    print(f"Built γ_t models: A(n={nA}), B(n={nB}).  <Σc^{-1}>: A={SC_A:.6e} kpc^2/Msun, B={SC_B:.6e} kpc^2/Msun")
if __name__=="__main__": main()
