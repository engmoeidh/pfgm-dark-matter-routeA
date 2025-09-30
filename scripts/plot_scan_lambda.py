import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results/tables/binC_scan_lambda.csv")
plt.figure(figsize=(6,4.5))
plt.semilogx(df["lam"], df["chi2"], '-o', ms=4)
plt.xlabel(r"$\lambda$ [Mpc]"); plt.ylabel(r"$\chi^2$")
plt.title("Bin-C: $\chi^2$ vs $\lambda$ (inner-fit A1h, b0, b2)")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("figures/lensing/bin_C_chi2_vs_lambda.png", dpi=160)
print("Wrote figures/lensing/bin_C_chi2_vs_lambda.png")
