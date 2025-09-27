import numpy as np
from scipy import integrate, special

def mu_lens(k, lam):
    """PFGM lensing filter μ_lens(k;λ) = 1/(1 + k^2 λ^2). k in 1/Mpc, λ in Mpc."""
    k = np.asarray(k, dtype=float)
    return 1.0 / (1.0 + (k*lam)**2)

def j2(x):
    """Spherical Bessel j2(x)."""
    return special.spherical_jn(2, x)

def hankel_j2(k, Pk, R):
    """
    Compute 2-halo-like Hankel integral with J2 kernel:
      I(R) = ∫_0^∞ dk k P(k) J2(kR) / (2π)
    Here we use spherical j2 to avoid oscillation issues for a quick demo.
    """
    # simple quadrature per R for a quick, robust baseline
    def integrand(k_, R_):
        return k_ * np.interp(k_, k, Pk) * j2(k_*R_) / (2.0*np.pi)
    I = np.zeros_like(R, dtype=float)
    for i, Rv in enumerate(np.atleast_1d(R)):
        I[i] = integrate.quad(lambda kk: integrand(kk, Rv), 0.0, k.max(), limit=300, epsabs=1e-6, epsrel=1e-4)[0]
    return I
