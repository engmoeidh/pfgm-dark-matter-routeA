import numpy as np
from scipy import integrate
from scipy.special import spherical_jn

# ---- lensing filter μ_lens(k; λ) = 1/(1 + k^2 λ^2) ----
def mu_lens(k, lam):
    k = np.asarray(k, dtype=float)
    return 1.0 / (1.0 + (k * lam) ** 2)

# ---- spherical Bessel j2(x) helper ----
def j2(x):
    return spherical_jn(2, x)

# ---- J2 Hankel via quad (accurate but slow) ----
def hankel_j2(k, Pk, R):
    """
    I(R) = ∫_0^kmax dk [ k P(k) j2(k R) ] / (2π)
    Accurate quad per R (slow) — useful for validations.
    """
    k = np.asarray(k, float)
    Pk = np.asarray(Pk, float)
    R = np.atleast_1d(R).astype(float)

    def integrand(kk, R_):
        return kk * np.interp(kk, k, Pk) * j2(kk * R_) / (2.0 * np.pi)

    out = np.zeros_like(R, dtype=float)
    kmax = float(k.max())
    for i, Rv in enumerate(R):
        out[i] = integrate.quad(lambda kk: integrand(kk, Rv),
                                0.0, kmax, limit=300, epsabs=1e-6, epsrel=1e-4)[0]
    return out

# ---- J2 Hankel via trapezoid (fast) ----
def hankel_j2_fast(k, Pk, R):
    """
    Fast projector on existing k-grid:
    I(R) ≈ ∫ dk [ k P(k) j2(k R) ] / (2π)  with trapezoid rule.
    Assumes k is monotone; Pk sampled on same grid.
    """
    k = np.asarray(k, float)
    Pk = np.asarray(Pk, float)
    R = np.atleast_1d(R).astype(float)
    w = (k * Pk) / (2.0 * np.pi)  # k-space weight
    out = np.empty_like(R, dtype=float)
    for i, Rv in enumerate(R):
        out[i] = np.trapz(w * spherical_jn(2, k * Rv), k)
    return out
