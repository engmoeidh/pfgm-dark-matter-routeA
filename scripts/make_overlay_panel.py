import os
import matplotlib.pyplot as plt
from PIL import Image

BINS  = ["A","B","C"]
MODES = ["freeA","A1","GR"]

def maybe_open(path):
    return Image.open(path).convert("RGB") if os.path.isfile(path) else None

def main():
    base = "figures/lensing"
    tiles = [[maybe_open(os.path.join(base, f"bin_{b}_overlay_{m}.png")) for m in MODES] for b in BINS]
    # get a reference tile size
    w = h = None
    for row in tiles:
        for im in row:
            if im: w,h = im.size; break
        if w: break
    if w is None:
        print("No overlays found."); return
    panel = Image.new("RGB", (3*w, 3*h), (255,255,255))
    for i,b in enumerate(BINS):
        for j,m in enumerate(MODES):
            im = tiles[i][j]
            if im: panel.paste(im, (j*w, i*h))
    os.makedirs(base, exist_ok=True)
    out_png = os.path.join(base,"overlays_panel_3x3.png")
    panel.save(out_png, "PNG")
    print("Wrote", out_png)
    # also PDF
    plt.figure(figsize=(3*w/100, 3*h/100), dpi=100)
    plt.axis("off")
    plt.imshow(panel)
    out_pdf = os.path.join(base,"overlays_panel_3x3.pdf")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    print("Wrote", out_pdf)

if __name__ == "__main__":
    main()
