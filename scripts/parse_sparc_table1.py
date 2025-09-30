import re, sys
import pandas as pd
from pathlib import Path

IN = Path("data/raw/sparc/SPARC_Table1_clean.csv")
OUT = Path("results/tables/SPARC_Table1_clean_parsed.csv")

def try_parsers(path: Path):
    # try CSV, then whitespace, then fixed-width
    # also attempt skipping preamble by detecting header with 'gal' token
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # find header line index (first line containing 'gal' token)
    hdr_idx = None
    for i, line in enumerate(text[:200]):
        if re.search(r'\bgal', line, flags=re.I):
            hdr_idx = i
            break
    skip = list(range(hdr_idx)) if hdr_idx is not None else []
    for kwargs in (dict(sep=",", engine="python", skiprows=skip),
                   dict(sep=r"\s+", engine="python", skiprows=skip)):
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 3:
                return df
        except Exception:
            pass
    # fixed width fallback
    try:
        df = pd.read_fwf(path, skiprows=skip)
        return df
    except Exception:
        raise

def main():
    df = try_parsers(IN)
    df.columns = [str(c).strip() for c in df.columns]

    # find galaxy column
    cand_gal = [c for c in df.columns if re.search(r'\b(galaxy|name)\b', c, flags=re.I)]
    gal = cand_gal[0] if cand_gal else df.columns[0]
    df = df.rename(columns={gal:"Galaxy"})
    df["Galaxy"] = df["Galaxy"].astype(str).str.strip()

    # drop rows that clearly aren't galaxies (TITLE, AUTHORS, etc.)
    bad = df["Galaxy"].str.match(r'^(TITLE|AUTHORS|BYTES|TABLE|NOTE|SUPP|ID)$', case=False, na=False)
    df = df[~bad].copy()

    # keep galaxy-like names (NGC/UGC/IC/DDO/ESO/M*/AGC*/KK*/KUG*/LSB*/F*D* etc.), but allow anything with letters+digits
    keep = df["Galaxy"].str.contains(r'[A-Za-z]', na=False)
    df = df[keep].copy()

    # quality column
    cand_Q = [c for c in df.columns if re.fullmatch(r'\s*Q\s*', c, flags=re.I)] or \
             [c for c in df.columns if re.search(r'qual', c, flags=re.I)]
    if cand_Q:
        Q = cand_Q[0]
        if Q != "Q": df = df.rename(columns={Q:"Q"})
        # coerce Q to int if possible
        try:
            df["Q"] = df["Q"].astype("Int64")
        except Exception:
            pass
    else:
        df["Q"] = 1

    # keep minimal subset
    out = df[["Galaxy","Q"]].dropna(subset=["Galaxy"]).drop_duplicates()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("Wrote", OUT, "with", len(out), "galaxies")

if __name__ == "__main__":
    main()
