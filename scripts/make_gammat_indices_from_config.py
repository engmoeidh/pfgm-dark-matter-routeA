import argparse, configparser, re, sys, numpy as np, pandas as pd
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pipeline", required=True)
p.add_argument("--values",   required=True)
p.add_argument("--cov",      required=True)
p.add_argument("--out_cov",  required=True)
p.add_argument("--out_index",required=True)
p.add_argument("--gt_pattern", default=r"(gammat|gamma_t|gt)")
a = p.parse_args()

# read configs
cp = configparser.ConfigParser(strict=False); cp.read(a.pipeline)
cv = configparser.ConfigParser(strict=False); cv.read(a.values)

# find the stats order line
stats_line=None
for sec in cp.sections():
    for k,v in cp[sec].items():
        if re.search(r"(stats|statistics|data_vector|twopt_vector|blocks?)", k, re.I):
            stats_line=v; break
    if stats_line: break
if not stats_line:
    sys.exit("No 'statistics' list in pipeline.ini; open and tell me the key to parse.")

# split into tokens (xip, xim, gammat, wtheta, etc.)
stats=[s for s in re.split(r"[,\s]+", stats_line) if s]
print("Statistics parsed:", stats)

# try to read per-block counts from values.ini (theta or bins)
def get_int(keypat):
    for sec in cv.sections():
        for k,v in cv[sec].items():
            if re.search(keypat, k, re.I):
                try: return int(float(v))
                except: pass
    return None

sizes={}
for st in stats:
    sizes[st]=( get_int(rf"{st}.*n(theta|bins)") or
                get_int(rf"n(theta|bins).*{st}") or
                get_int(rf"{st}_n") )

C=np.loadtxt(a.cov, dtype=float); N=C.shape[0]
known=sum([sizes[s] for s in stats if sizes.get(s) is not None]) if any(sizes.values()) else 0
unknown=[s for s in stats if sizes.get(s) is None]
rem=N-known
if rem<0: sys.exit(f"Block sizes exceed covariance length: {known}>{N}")
if unknown:
    share= rem//len(unknown) if len(unknown)>0 else 0
    for s in unknown[:-1]: sizes[s]=share
    sizes[unknown[-1]]= N - sum([sizes[s] for s in stats if sizes.get(s) is not None])
print("Resolved sizes:", sizes, " sum=", sum(sizes.values()))

# cumulative index
start=0; spans={}
for st in stats:
    n=sizes[st]; spans[st]=(start, start+n); start+=n

gt_key=None
for st in stats:
    if re.search(a.gt_pattern, st, re.I): gt_key=st; break
if gt_key is None:
    sys.exit("Could not identify gammat block; adjust --gt_pattern")

i0,i1=spans[gt_key]
Cg=C[i0:i1, i0:i1]
Path(a.out_cov).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(Cg).to_csv(a.out_cov, header=False, index=False)
with open(a.out_index, "w", encoding="utf-8") as f:
    f.write(f"# {gt_key} indices 0-based: {i0}:{i1}\n")
print(f"Wrote γ_t covariance block: {a.out_cov}  shape={Cg.shape}")
print(f"Wrote index file: {a.out_index}")
