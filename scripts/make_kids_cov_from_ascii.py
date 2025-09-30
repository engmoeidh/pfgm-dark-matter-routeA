import argparse, numpy as np, pandas as pd
from pathlib import Path
import glob, sys

def read_ascii_matrix(p: Path):
    # robust whitespace reader; ignores empty/comment lines
    arr = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        parts = line.replace(",",".").split()
        try:
            arr.append([float(x) for x in parts])
        except Exception:
            pass
    M = np.array(arr, dtype=float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1]:
        raise ValueError(f"Not a square matrix: {p} -> {M.shape}")
    return M

def pick_by_size(folder, N):
    cands=[]
    for p in Path(folder).rglob("*.ascii"):
        try:
            # quick peek first row count without loading all lines
            M = read_ascii_matrix(p)
            if M.shape==(N,N): cands.append(p)
        except Exception:
            continue
    return cands

def save_cov_csv(M, outpath: Path, tag: str):
    # symmetrize and PD-guard
    M = 0.5*(M+M.T)
    # tiny diagonal regularization if needed
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        eps = 1e-6*np.median(np.diag(M))
        M = M + eps*np.eye(M.shape[0])
        print(f"[{tag}] added diag regularization: {eps:.3e}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M).to_csv(outpath, header=False, index=False)
    print(f"[{tag}] wrote {outpath}  shape={M.shape}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing ASCII covariance files (*.ascii)")
    ap.add_argument("--kidsA", required=True, help="Repo KiDS bin-A CSV (R, ΔΣ, σ)")
    ap.add_argument("--kidsB", required=True, help="Repo KiDS bin-B CSV (R, ΔΣ, σ)")
    ap.add_argument("--outA", default="data/raw/kids/KiDS_binA_cov.csv")
    ap.add_argument("--outB", default="data/raw/kids/KiDS_binB_cov.csv")
    args=ap.parse_args()

    NA = len(pd.read_csv(args.kidsA))
    NB = len(pd.read_csv(args.kidsB))

    cA = pick_by_size(args.src, NA)
    cB = pick_by_size(args.src, NB)

    if not cA:
        sys.exit(f"No ASCII covariance of size {NA}x{NA} found under {args.src}")
    if not cB:
        sys.exit(f"No ASCII covariance of size {NB}x{NB} found under {args.src}")

    # if multiple candidates, pick the first and print suggestions
    if len(cA)>1: print("[A] multiple candidates, picking:", cA[0], "\n  others:", *cA[1:], sep="\n  ")
    if len(cB)>1: print("[B] multiple candidates, picking:", cB[0], "\n  others:", *cB[1:], sep="\n  ")

    MA = read_ascii_matrix(cA[0])
    MB = read_ascii_matrix(cB[0])

    save_cov_csv(MA, Path(args.outA), "A")
    save_cov_csv(MB, Path(args.outB), "B")

if __name__=="__main__":
    main()
