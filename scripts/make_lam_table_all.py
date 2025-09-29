import json, glob, os

def edge_fmt(val, lb=0.05, ub=5.0):
    if val is None: return "--"
    try:
        x=float(val)
        if abs(x-ub) < 0.02*(ub-lb): return r"$> \mathrm{few\ Mpc}$"
        if abs(x-lb) < 0.02*(ub-lb): return f"$\\approx {lb:.2g}$"
        return f"{x:.3g}"
    except: return str(val)

def grab(js):
    d=json.load(open(js,"r"))
    return dict(
        bin=d.get("bin"),
        mode=d.get("mode"),
        lam=d.get("lam"),
        chi2nu=d.get("chi2/nu"),
        AIC=d.get("AIC"),
        BIC=d.get("BIC"),
        dAIC=d.get("ΔAIC_vs_GR"),
        dBIC=d.get("ΔBIC_vs_GR"),
    )

files=sorted(glob.glob("results/tables/bin_*_fit_*.json"))
rows=[grab(f) for f in files]
# Order modes and bins for readability
mode_order={"GRkernel":0,"freeA":1,"A1":2}
bin_order={"A":0,"B":1,"C":2,"LOWZ":3,"CMASS":4}
rows.sort(key=lambda r:(bin_order.get(r["bin"],99), mode_order.get(r["mode"],99)))

print(r"\begin{table}[t]")
print(r"\centering")
print(r"\small")
print(r"\begin{tabular}{l l r r r r r}")
print(r"\toprule")
print(r"Bin & Mode & $\lambda$ [Mpc] & $\chi^2/\nu$ & AIC & BIC & $\Delta$AIC (vs GR) \\")
print(r"\midrule")
curbin=None
for r in rows:
    if r["bin"]!=curbin and curbin is not None:
        print(r"\midrule")
    curbin=r["bin"]
    lam = edge_fmt(r["lam"])
    chn = f"{r['chi2nu']:.3g}" if r["chi2nu"] is not None else "--"
    aic = f"{r['AIC']:.3g}"     if r["AIC"]     is not None else "--"
    bic = f"{r['BIC']:.3g}"     if r["BIC"]     is not None else "--"
    dA  = "--" if r["mode"]=="GRkernel" else (f"{r['dAIC']:.3g}" if r["dAIC"] is not None else "--")
    print(f"{r['bin']} & {r['mode']} & {lam} & {chn} & {aic} & {bic} & {dA} \\\\")
print(r"\bottomrule")
print(r"\caption{Model comparison across KiDS (A,B,C) and SDSS (LOWZ, CMASS). Modes: GRkernel ($\lambda=0$), freeA, and A1-locked. $\Delta$AIC is relative to GR per bin. Values at the prior edge are reported as lower limits.}")
print(r"\label{tab:lam_all}")
print(r"\end{table}")
