import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)   # not used in this scaffold, kept for CLI compatibility
    ap.add_argument("--summary", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--appendix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.pdf), exist_ok=True)
    os.makedirs(os.path.dirname(args.appendix), exist_ok=True)

    # Load summary (arbitrary columns ok)
    df = pd.read_csv(args.summary)
    # Save appendix (verbatim copy/canonicalized order)
    df.to_csv(args.appendix, index=False)

    # Build a 1–2 page PDF using matplotlib's table (first ~30 rows for legibility)
    subset = df.head(30)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title("PFGM Guards Manifesto — Summary (first 30 rows)", fontsize=14, pad=12)
    tbl = ax.table(cellText=subset.values,
                   colLabels=subset.columns.tolist(),
                   loc="upper left", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.2)
    plt.tight_layout()
    plt.savefig(args.pdf, dpi=200)
    plt.close()
    print("Wrote", args.pdf, "and", args.appendix)

if __name__ == "__main__":
    main()
