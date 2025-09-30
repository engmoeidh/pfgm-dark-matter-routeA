import argparse, re, numpy as np, pandas as pd
from pathlib import Path

def read_ascii(p: Path):
    rows=[]
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln=ln.strip()
        if not ln or ln.startswith("#"): continue
        # tolerate commas
        parts = ln.replace(",",".").split()
        try:
            rows.append([float(x) for x in parts])
        except:
            pass
    M = np.array(rows, dtype=float)
    return M

def is_square(M, N): return (M.ndim==2) and (M.shape==(N,N))

def try_load(p: Path, N: int):
    try:
        M = read_ascii(p)
        return M if is_square(M,N) else None
    except Exception:
        return None

def search_candidates(sources, N):
    # prefer filenames with nBins{N} token; else fall back to any square {N}x{N}
    token = re.compile(fr"nBins\s*{N}(?=\D|$)", re.I)
    hits_tok, hits_shape = [], []
    exts = (".ascii",".dat",".txt")
    for src in sources:
        for p in Path(src).rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts: continue
            if token.search(p.name):
                M = try_load(p,N)
                if M is not None: hits_tok.append((p,M))
            else:
                M = try_load(p,N)
                if M is not None: hits_shape.append((p,M))
    return hits_tok or hits_shape

def make_surrogate_cov(kids_csv, ell=0.7):
    # diagonal-sigma kernel with exp(-|lnR_i-lnR_j|/ell)
    d  = pd.read_csv(kids_csv).to_numpy(float)
    R, E = d[:,0], d[:,2]
    lnR  = np.log(np.clip(R,1e-12,None))
    C = np.empty((len(R),len(R)))
    for i in range(len(R)):
        for j in range(len(R)):
            rho = np.exp(-abs(lnR[i]-lnR[j])/float(ell))
            C[i,j] = rho * (E[i]*E[j])
    return C

def write_cov(M, out: Path, tag: str):
    # symmetrize + tiny PD regularization
    M = 0.5*(M+M.T)
    try:
        np.linalg.cholesky(M)
        eps_used = 0.0
    except np.linalg.LinAlgError:
        eps = 1e-6*np.median(np.diag(M))
        M += eps*np.eye(M.shape[0])
        eps_used = eps
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M).to_csv(out, header=False, index=False)
    print(f"[{tag}] wrote {out}  shape={M.shape}  reg={eps_used:.3e}")
    return M

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="1 or more folders to scan recursively for ASCII covariances")
    ap.add_argument("--kidsA", required=True)
    ap.add_argument("--kidsB", required=True)
    ap.add_argument("--outA", default="data/raw/kids/KiDS_binA_cov.csv")
    ap.add_argument("--outB", default="data/raw/kids/KiDS_binB_cov.csv")
    ap.add_argument("--ell",  type=float, default=0.7, help="surrogate kernel correlation length in ln R")
    args=ap.parse_args()

    NA = len(pd.read_csv(args.kidsA))
    NB = len(pd.read_csv(args.kidsB))

    # Search for A
    cA = search_candidates(args.src, NA)
    if cA:
        pA, MA = cA[0]
        print(f"[A] using {pA}")
    else:
        print(f"[A] no matching file found; building SURROGATE from σ with ell={args.ell}")
        MA = make_surrogate_cov(args.kidsA, args.ell)

    # Search for B
    cB = search_candidates(args.src, NB)
    if cB:
        pB, MB = cB[0]
        print(f"[B] using {pB}")
    else:
        print(f"[B] no matching file found; building SURROGATE from σ with ell={args.ell}")
        MB = make_surrogate_cov(args.kidsB, args.ell)

    write_cov(MA, Path(args.outA), "A")
    write_cov(MB, Path(args.outB), "B")

if __name__=="__main__":
    main()
