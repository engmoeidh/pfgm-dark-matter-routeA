import argparse, numpy as np, pandas as pd, re, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cov", required=True)     # square full covariance (e.g. 142x142)
p.add_argument("--labels", required=True)  # one line per element of data vector, with a name/tag
p.add_argument("--out", required=True)     # CSV for γ_t submatrix
p.add_argument("--pattern", default=r"(gammat|gamma_t|gt)", help="regex to match γ_t entries")
a = p.parse_args()

C = np.loadtxt(a.cov, dtype=float)
if C.ndim!=2 or C.shape[0]!=C.shape[1]:
    sys.exit(f"ERROR: covariance not square {C.shape}")

raw = [ln.strip() for ln in open(a.labels, encoding="utf-8", errors="ignore") if ln.strip()]
# label key: first token with letters or underscore
labs=[]
for s in raw:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\-\.:]*", s)
    labs.append(toks[0] if toks else s)

if len(labs)!=C.shape[0]:
    sys.exit(f"ERROR: labels length {len(labs)} != cov dim {C.shape[0]}")

idx = [i for i,s in enumerate(labs) if re.search(a.pattern, s, re.I)]
if not idx:
    sys.exit("ERROR: no γ_t entries matched; try a different --pattern")

idx = np.asarray(idx, int)
Cg = C[np.ix_(idx, idx)]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(Cg).to_csv(a.out, header=False, index=False)

meta = Path(a.out).with_suffix(".index.txt")
with open(meta, "w", encoding="utf-8") as g:
    g.write("# 0-based indices of γ_t in the full data vector\n")
    g.write(",".join(map(str, idx.tolist()))+"\n")
    g.write("# first 15 matched labels:\n")
    for k in idx[:15]: g.write(labs[k]+"\n")

print(f"Wrote γ_t block: {a.out}  shape={Cg.shape}")
print(f"Wrote index: {meta}")
