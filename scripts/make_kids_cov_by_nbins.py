import argparse, re, numpy as np, pandas as pd
from pathlib import Path

def read_ascii_matrix(p: Path):
    rows=[]
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln=ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = ln.replace(",", ".").split()
        try: rows.append([float(x) for x in parts])
        except: pass
    M = np.array(rows, dtype=float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1]:
        raise ValueError(f"Not square: {p} -> {M.shape}")
    return M

def best_by_nbins(folder: Path, N: int):
    # Prefer files whose name contains nBins{N}; else fall back to any {N}x{N}
    hits_token = [p for p in folder.rglob("*.ascii") if re.search(fr"nBins\s*{N}(?=\D|$)", p.name, re.I)]
    if hits_token: return hits_token[0]
    # fallback by shape
    for p in folder.rglob("*.ascii"):
        try:
            M = read_ascii_matrix(p)
            if M.shape==(N,N): return p
        except: pass
    return None

def write_cov_csv(M: np.ndarray, out: Path, tag: str):
    M = 0.5*(M+M.T)
    # small regularization if not PD
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        eps = 1e-6*np.median(np.diag(M))
        M += eps*np.eye(M.shape[0])
        print(f"[{tag}] added diag regularization: {eps:.3e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M).to_csv(out, header=False, index=False)
    print(f"[{tag}] wrote {out}  shape={M.shape}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with *.ascii covariances")
    ap.add_argument("--kidsA", required=True)   # repo KiDS A data
    ap.add_argument("--kidsB", required=True)   # repo KiDS B data
    ap.add_argument("--outA", default="data/raw/kids/KiDS_binA_cov.csv")
    ap.add_argument("--outB", default="data/raw/kids/KiDS_binB_cov.csv")
    args=ap.parse_args()

    kidsA = pd.read_csv(args.kidsA)
    kidsB = pd.read_csv(args.kidsB)
    NA, NB = len(kidsA), len(kidsB)

    src = Path(args.src)
    pA = best_by_nbins(src, NA)
    pB = best_by_nbins(src, NB)
    if pA is None:
        raise SystemExit(f"No covariance matching nBins{NA} (or {NA}x{NA}) under {src}")
    if pB is None:
        raise SystemExit(f"No covariance matching nBins{NB} (or {NB}x{NB}) under {src}")

    MA = read_ascii_matrix(pA); MB = read_ascii_matrix(pB)
    write_cov_csv(MA, Path(args.outA), "A")
    write_cov_csv(MB, Path(args.outB), "B")

if __name__=="__main__":
    main()
