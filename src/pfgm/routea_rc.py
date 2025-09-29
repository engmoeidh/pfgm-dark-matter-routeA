import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Structural scales loader ----------
def load_struct_scales(table_path="data/raw/sparc/SPARC_Table1_clean.csv"):
    """
    Returns a dict: Galaxy -> dict(Rd, Rb, Rg) in kpc when available.
    Uses SPARC Table1 columns if present; otherwise entries are None.
    """
    p = Path(table_path)
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    cols = {c.lower().strip(): c for c in df.columns}
    name = cols.get("galaxy", None)
    Rd   = cols.get("rdisk", None)  # disk exponential scale length [kpc]
    Reff = cols.get("reff",  None)  # bulge/spheroid effective radius [kpc] (if present)
    RHI  = cols.get("rhi",   None)  # HI radius at 1 Msun/pc^2 [kpc]

    out = {}
    for _,row in df.iterrows():
        g = str(row[name]) if name else None
        if not g: continue
        out[g] = dict(
            Rd = float(row[Rd]) if Rd and pd.notna(row[Rd]) else None,
            Rb = float(row[Reff]) if Reff and pd.notna(row[Reff]) else None,
            Rg = float(row[RHI]) if RHI and pd.notna(row[RHI]) else None,
        )
    return out

# ---------- Route-A delta-V^2 scaffold ----------
def routea_delta_v2(R, Vdisk, Vbulge, Vgas, Rd=None, Rb=None, Rg=None, lam_kpc=5.0, eps=0.2):
    """
    Scaffold for the geodesic correction produced by Route-A in the weak field.
    The form below is a fast surrogate:
      δV^2(R) = eps * [ w_d * f(R,Rd,lam) + w_b * f(R,Rb,lam) + w_g * f(R,Rg,lam) ] * (Vdisk^2 + Vbulge^2 + Vgas^2)
    where f encodes Yukawa-like range weighting and w_* normalize by the relative baryonic contributions.
    Replace f() or the whole body by your exact kernel when ready.
    """
    R = np.asarray(R, float)
    Vd2 = (Vdisk if Vdisk is not None else 0.0)**2
    Vb2 = (Vbulge if Vbulge is not None else 0.0)**2
    Vg2 = (Vgas  if Vgas  is not None else 0.0)**2
    Vbar2 = Vd2 + Vb2 + Vg2 + 1e-30

    # Relative weights by fraction of Vbar^2 (freeze at small radii)
    wd = Vd2 / Vbar2
    wb = Vb2 / Vbar2
    wg = Vg2 / Vbar2

    # Range weight function: larger contribution where component is concentrated.
    def f_comp(R, Rc):
        if Rc is None or Rc <= 0.0:
            Rc = 1.0
        # Smooth kernel with two scales (component Rc and geodesic λ):
        x = R / max(Rc, 1e-3)
        y = R / max(lam_kpc, 1e-3)
        # bounded, positive, decays for large R and saturates for R >> Rc but R << λ
        return (x / (1.0 + x)) * (1.0 - np.exp(-y))

    fd = f_comp(R, Rd)
    fb = f_comp(R, Rb)
    fg = f_comp(R, Rg)

    # Put together; scale by Vbar^2 to keep units consistent.
    dV2 = eps * (wd * fd + wb * fb + wg * fg) * Vbar2
    return dV2

def routea_rc_model_sq(R_kpc, Vdisk, Vbulge, Vgas, scales, lam_kpc, eps):
    """
    V_mod^2(R) = V_bar^2(R) + δV^2_RouteA(R).
    V* arrays are in km/s; scales is dict with (Rd,Rb,Rg) in kpc.
    """
    Rd = scales.get("Rd")
    Rb = scales.get("Rb")
    Rg = scales.get("Rg")
    Vd2 = (Vdisk if Vdisk is not None else 0.0)**2
    Vb2 = (Vbulge if Vbulge is not None else 0.0)**2
    Vg2 = (Vgas  if Vgas  is not None else 0.0)**2
    Vbar2 = Vd2 + Vb2 + Vg2
    dV2   = routea_delta_v2(R_kpc, Vdisk, Vbulge, Vgas, Rd=Rd, Rb=Rb, Rg=Rg, lam_kpc=lam_kpc, eps=eps)
    return Vbar2 + dV2
