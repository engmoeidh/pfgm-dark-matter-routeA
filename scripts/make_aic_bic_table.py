import json, glob

rows = []
for f in sorted(glob.glob("results/tables/bin_*_fit_*.json")):
    d = json.load(open(f,"r"))
    rows.append({
        "bin": d["bin"],
        "mode": d["mode"],
        "lam": d.get("lam"),
        "chi2": d.get("chi2"),
        "nu": d.get("nu"),
        "chi2nu": d.get("chi2/nu"),
        "AIC": d.get("AIC"),
        "BIC": d.get("BIC"),
        "dAIC": d.get("ΔAIC_vs_GR"),
        "dBIC": d.get("ΔBIC_vs_GR"),
    })

# group by bin; modes we care about in order:
order = ["GRkernel","freeA","A1"]


def fmt_lambda(x, lb=0.05, ub=5.0):
    if x is None: return "--"
    try:
        x = float(x)
        # mark near bounds (within 2% of range)
        if abs(x-ub) < 0.02*(ub-lb): return r"$>\!\mathrm{few\ Mpc}$"
        if abs(x-lb) < 0.02*(ub-lb): return f"$\approx$ {lb:.2g}"
        return f"{x:.3g}"
    except Exception:
        return str(x)

def ffmt(x, digs=3):
    if x is None: return "--"
    try:
        return f"{x:.{digs}g}"
    except Exception:
        return str(x)

print(r"\begin{table}[t]")
print(r"\centering")
print(r"\small")
print(r"\begin{tabular}{l l r r r r r}")
print(r"\toprule")
print(r"Bin & Mode & $\lambda$ [Mpc] & $\chi^2/\nu$ & AIC & BIC & $\Delta$AIC (vs GR) \\")
print(r"\midrule")
bybin = {}
for r in rows:
    bybin.setdefault(r["bin"], {})[r["mode"]] = r

for b in sorted(bybin):
    grp = bybin[b]
    for m in order:
        if m not in grp: continue
        r = grp[m]
        lam = fmt_lambda(r["lam"])
        chn = ffmt(r["chi2nu"])
        aic = ffmt(r["AIC"])
        bic = ffmt(r["BIC"])
        dA  = ffmt(r["dAIC"]) if m!="GRkernel" else "--"
        line = f"{b} & {m} & {lam} & {chn} & {aic} & {bic} & {dA} \\\\"
        print(line)
    print(r"\midrule")
print(r"\bottomrule")
print(r"\caption{Model comparison for KiDS bins A and B. Modes: GRkernel ($\lambda=0$), freeA, and A1-locked ($A_{\rm 1h}=1$). $\Delta$AIC is relative to GR per bin.}")
print(r"\label{tab:aic_bic_bins}")
print(r"\end{table}")
