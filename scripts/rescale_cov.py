import argparse, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--cov", required=True)
p.add_argument("--scale", type=float, required=True)
a = p.parse_args()
C = pd.read_csv(a.cov, header=None).to_numpy(float)
C *= a.scale
pd.DataFrame(C).to_csv(a.cov, header=False, index=False)
print(f"Rescaled {a.cov} by {a.scale:g}")
