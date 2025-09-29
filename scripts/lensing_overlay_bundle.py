import argparse, os, glob
from pathlib import Path
from PIL import Image, ImageOps

def collect_pngs(base_dir, survey):
    base = Path(base_dir)
    pat = "bin*_overlay_*.png" if survey=="KiDS" else "*overlay*.png"
    return sorted([p for p in (base/"figures"/"lensing").glob(pat)])

def tile(images, out_path, grid):
    rows, cols = grid
    # ensure we have exactly rows*cols items (pad with last)
    if not images:
        img = Image.new("RGB", (900,600), (255,255,255))
        ImageOps.expand(img, border=2).save(out_path)
        return
    imgs = [Image.open(p).convert("RGB") for p in images]
    while len(imgs) < rows*cols:
        imgs.append(imgs[-1])
    W = max(i.width for i in imgs)
    H = max(i.height for i in imgs)
    canvas = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for idx, im in enumerate(imgs[:rows*cols]):
        r, c = divmod(idx, cols)
        canvas.paste(im.resize((W,H)), (c*W, r*H))
    canvas.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kids-json", help="Not used by this scaffold; kept for CLI compatibility")
    ap.add_argument("--sdss-json", help="Not used by this scaffold")
    ap.add_argument("--out-panel", required=True)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    os.makedirs(Path(args.out_panel).parent, exist_ok=True)

    # detect which survey from filename
    grid = (3,3) if "3x3" in args.out_panel or "KiDS" in args.zip else (2,3)
    survey = "KiDS" if grid==(3,3) else "SDSS"
    imgs = collect_pngs(Path("."), survey)
    if not imgs:
        # fallback to any lensing PNGs
        imgs = sorted(Path("figures/lensing").glob("*.png"))
    tile(imgs, args.out_panel, grid)

    # zip bundle
    import zipfile
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in imgs:
            zf.write(p, arcname=p.name)
    print("Wrote", args.out_panel, "and", args.zip, f"({len(imgs)} images)")
if __name__ == "__main__":
    main()
