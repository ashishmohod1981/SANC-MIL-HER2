"""Fig. 6 (example maps) rebuilt for print: the seven-column map panel produced by the pipeline (fig08_localization_maps.png) is cut
into its 28 cells, the pipeline's own small labels are removed, and titles, row labels and colour bars are re-set at >= 8 pt."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import sys, numpy as np
sys.path.insert(0, HERE)
from figstyle import STYLE, enforce, report, OUTDIR
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
plt.rcParams.update(STYLE)
src = np.asarray(Image.open(_os.path.join(ROOT, "results", "gea", "figures", "fig08_localization_maps.png")).convert("RGB")).copy()
cols = [(30, 674), (714, 1358), (1398, 2042), (2083, 2727), (2767, 3411), (3451, 4095), (4135, 4779)]
rows = [(80, 724), (761, 1405), (1442, 2086), (2123, 2767)]
for (y0, y1) in rows:                                   # erase the pipeline's in-panel row label (top-left of column 1)
    x0 = cols[0][0]; bg = src[y0 + 70, x0 + 4].copy(); src[y0:y0 + 48, x0:x0 + 340] = bg
titles = ["Core", "DAB optical\ndensity", "Membranous DAB\n≥ moderate", "Grad-CAM\n(per-core norm.)", "Attention MIL\nweight", "SANC-MIL\nP(tile positive)", "SANC-MIL SD\n(5 fold models)"]
rlab = ["IHC 0\nP = 0.01", "2+/FISH−\nP = 0.32", "2+/FISH+\nP = 1.00", "IHC 3+\nP = 1.00"]
specs = [None, ("magma", 0, 1.614, "OD"), ("magma", None, None, "per core"), ("viridis", 0, 1, "per core"),
         ("viridis", 0, 1, "÷ core max"), ("RdBu_r", 0, 1, "probability"), ("Greys", 0, 0.35, "SD")]
W, L, T, B = 6.85, 0.62, 0.40, 0.62
cw = (W - L - 0.04) / 7; H = T + 4 * cw + B
fig = plt.figure(figsize=(W, H))
for r, (y0, y1) in enumerate(rows):
    for c, (x0, x1) in enumerate(cols):
        ax = fig.add_axes([(L + c * cw + 0.01) / W, (B + (3 - r) * cw + 0.01) / H, (cw - 0.02) / W, (cw - 0.02) / H])
        ax.imshow(src[y0:y1, x0:x1]); ax.axis("off")
        if r == 0: ax.set_title(titles[c], fontsize=8, pad=3, linespacing=1.1)
        if r == 3 and c == 0:                                # scale bar: the panel shows the whole core thumbnail, 5,468 px x 0.23 um = 1,258 um across
            px = (x1 - x0) * 250.0 / 1258.0
            ax.plot([30, 30 + px], [(y1 - y0) - 40] * 2, color="black", lw=2.0, solid_capstyle="butt")
            ax.text(30 + px / 2, (y1 - y0) - 62, "250 \u00b5m", ha="center", va="bottom", fontsize=8)
    fig.text(0.04 / W, (B + (3 - r) * cw + cw / 2) / H, rlab[r], fontsize=8, va="center", ha="left", linespacing=1.2)
for c, sp in enumerate(specs):
    if sp is None: continue
    cm, lo, hi, lab = sp
    cax = fig.add_axes([(L + c * cw + 0.08) / W, 0.40 / H, (cw - 0.16) / W, 0.07 / H])
    if lo is None:
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cm), cax=cax, orientation="horizontal"); cb.set_ticks([0, 1]); cb.set_ticklabels(["0", "max"])
    else:
        cb = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cm), cax=cax, orientation="horizontal"); cb.set_ticks([lo, hi]); cb.set_ticklabels([f"{lo:g}", f"{hi:.2g}"])
    cb.ax.tick_params(labelsize=8, length=2, width=0.5); cb.outline.set_linewidth(0.5)
    cax.set_xlabel(lab, fontsize=8, labelpad=1)
enforce(fig); print("Fig5_maps", report(fig))
for ext in ("png", "pdf"): fig.savefig(OUTDIR + f"Fig5_maps.{ext}", dpi=600 if ext == "png" else 400, facecolor="white")
