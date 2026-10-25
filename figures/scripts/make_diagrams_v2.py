"""Publication diagrams (Figs. 1-3) for the MBEC submission, final print size 174 mm wide, lettering >= 8 pt.
Card style: coloured header band with white title, light body. Every block, size and flow is taken from the v2.1 code
(p02-p11) and the analysis cells; nothing is drawn that the pipeline does not do."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image

OUT = _os.path.join(ROOT, "figures", "output") + "/"
CORE = _os.path.join(ROOT, "figures", "assets", "example_core.jpg")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "Liberation Serif:italic",
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600})
W = 6.85
INK, GREY, LINE = "#1d1d1f", "#5f6368", "#4a4f57"
PAL = {  # header colour, body fill
    "data": ("#2563a8", "#eef4fb"), "fixed": ("#6b6e73", "#f3f3f1"), "train": ("#16805a", "#ebf6f0"),
    "out": ("#5a46b8", "#f1eefb"), "loss": ("#c4501f", "#fdf0ea"), "eval": ("#a8730a", "#fdf6e5"), "ext": ("#9b3d74", "#fbeff5")}

def canvas(h):
    fig = plt.figure(figsize=(W, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, h); ax.axis("off"); return fig, ax

def card(ax, x, y, w, h, kind, title, body=None, hb=0.22, fs=8, dashed=False, align="center", bx=None):
    hc, fc = PAL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=hc, lw=0.9,
                                ls=(0, (3.5, 2)) if dashed else "-", zorder=2))
    if title:
        ax.add_patch(FancyBboxPatch((x, y + h - hb), w, hb, boxstyle="round,pad=0,rounding_size=0.06", fc=hc, ec=hc, lw=0.9, zorder=3))
        ax.add_patch(Rectangle((x + 0.004, y + h - hb), w - 0.008, hb / 2, fc=hc, ec="none", zorder=3))      # square lower corners of band
        ax.text(x + w / 2, y + h - hb / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="white", zorder=4)
    if body:
        top = y + h - (hb if title else 0)
        tx = (bx if bx is not None else x + w / 2) if align == "center" else x + 0.08
        ax.text(tx, (top + y) / 2, body, ha=align, va="center", fontsize=fs, color=INK, linespacing=1.3, zorder=4)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)

def frame(ax, x, y, w, h, label, color="#9aa0a6", fs=8, lx=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="none", ec=color, lw=0.8, ls=(0, (4, 2.5)), zorder=1))
    t = ax.text(x + lx, y + h, " " + label + " ", ha="left", va="center", fontsize=fs, fontweight="bold", color=color, zorder=5)
    t.set_bbox(dict(fc="white", ec="none", pad=0.5))

def path_arrow(ax, pts, color=LINE, lw=1.0, dashed=False, head=True):
    """Orthogonal connector with rounded corners."""
    verts, codes, r = [pts[0]], [Path.MOVETO], 0.07
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        d1 = np.hypot(x1 - x0, y1 - y0); d2 = np.hypot(x2 - x1, y2 - y1); rr = min(r, d1 / 2, d2 / 2)
        a = (x1 - (x1 - x0) / d1 * rr, y1 - (y1 - y0) / d1 * rr); b = (x1 + (x2 - x1) / d2 * rr, y1 + (y2 - y1) / d2 * rr)
        verts += [a, (x1, y1), b]; codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    if head:
        (xa, ya), (xb, yb) = verts[-1], pts[-1]; d = np.hypot(xb - xa, yb - ya); k = 0.055 / d if d > 0 else 0
        end = (xb - (xb - xa) * k, yb - (yb - ya) * k)
    else:
        end = pts[-1]
    verts.append(end); codes.append(Path.LINETO)
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=color, lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=1.5, capstyle="butt"))
    if head:
        ax.add_patch(FancyArrowPatch(verts[-1], pts[-1], arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=1.6))

def label(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 8); kw.setdefault("color", GREY); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    t = ax.text(x, y, s, zorder=6, **kw); t.set_bbox(dict(fc="white", ec="none", pad=0.6)); return t

def tma_icon(ax, cx, cy, r=0.17, color="#2563a8"):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=0.8, zorder=5))
    for dx, dy in [(-0.07, 0.07), (0.0, 0.07), (0.07, 0.07), (-0.07, 0.0), (0.0, 0.0), (0.07, 0.0), (-0.07, -0.07), (0.0, -0.07), (0.07, -0.07)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.028, fc="#c7d7ec", ec="none", zorder=6))

def slide_icon(ax, cx, cy, color="#9b3d74"):
    ax.add_patch(FancyBboxPatch((cx - 0.2, cy - 0.08), 0.4, 0.16, boxstyle="round,pad=0,rounding_size=0.02", fc="white", ec=color, lw=0.8, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - 0.06, cy - 0.055), 0.2, 0.11, boxstyle="round,pad=0,rounding_size=0.03", fc="#ecc9dc", ec="none", zorder=6))

def save(fig, name):
    for ext in ("pdf", "png"): fig.savefig(OUT + f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)

# ============================================================================ Fig. 1  Study design
import matplotlib.patheffects as pe
H1 = 4.80
fig, ax = canvas(H1)
HX = 0.32                                    # row headings start right of the left-hand flow line
ax.text(HX, H1 - 0.05, "Data", fontsize=9, fontweight="bold", color=INK, va="top")
dy, dh = 3.70, 0.84
d1 = card(ax, 0.06, dy, 2.18, dh, "data", "GEA training set", "1,598 TMA cores (1,554 used)\nIHC 0-3; FISH for 2+ cores\nno patient IDs, no pixel masks", bx=0.06 + (0.46 + 2.18) / 2)
d2 = card(ax, 2.34, dy, 2.08, dh, "data", "GEA test set", "307 cores, later patients\n16 HER2-positive\nheld out until the end", bx=2.34 + (0.46 + 2.08) / 2)
d3 = card(ax, 4.52, dy, 2.27, dh, "ext", "ADEL breast cohort", "418 slides, IHC 0-3\nother centre, scanner, organ\npixel size not recorded", bx=4.52 + (0.52 + 2.27) / 2)
for d in (d1, d2): tma_icon(ax, d["l"] + 0.25, d["b"] + 0.31)
slide_icon(ax, d3["l"] + 0.28, d3["b"] + 0.31)

ax.text(HX, 3.52, "Development (training cores only)", fontsize=9, fontweight="bold", color=INK, va="top")
ry, rh = 2.30, 0.92
c1 = card(ax, 0.06, ry, 1.24, rh, "fixed", "Stain calibration", "Macenko matrix\n(160 cores); DAB\nthresholds, IHC 0")
c2 = card(ax, 1.42, ry, 1.24, rh, "fixed", "Tiles, features", "frozen ResNet-50\nembedding + 14\nstain features")
frame(ax, 2.90, ry - 0.09, 1.98, rh + 0.25, "Grouped 5-fold CV", color="#16805a")
c3 = card(ax, 3.00, ry, 1.78, rh, "train", "SANC-MIL", "+ 9 comparators, 6 ablations\n527 proxy patients; 3 seeds\nT and thresholds from OOF")
c4 = card(ax, 5.00, ry, 1.56, rh, "train", "Membrane student", "pseudo-labels from the\nrule detector + OOF tile\nscores \u2192 segmentation net")
path_arrow(ax, [(0.17, d1["b"]), (0.17, c1["t"])])
path_arrow(ax, [(c1["r"], c1["cy"]), (c2["l"], c1["cy"])])
path_arrow(ax, [(c3["r"], c3["cy"]), (c4["l"], c4["cy"])])
# test set: one straight dashed line through the gap between feature extraction and cross-validation
TX = 2.77
path_arrow(ax, [(TX, d2["b"]), (TX, 1.38)], dashed=True, color="#a8730a")
# feature arrow drawn on top with a white halo so the crossing reads as a bridge
hl = FancyArrowPatch((c2["r"], c2["cy"]), (c3["l"], c2["cy"]), arrowstyle="-|>", mutation_scale=8, color=LINE, lw=1.0, shrinkA=0, shrinkB=0, zorder=7)
hl.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]); ax.add_patch(hl)
label(ax, TX, 1.98, "scored\nonce", fontsize=8, color="#a8730a")

ax.text(HX, 1.86, "Evaluation", fontsize=9, fontweight="bold", color=INK, va="top")
ey, eh = 0.08, 1.02
frame(ax, 0.06, ey - 0.03, 3.86, eh + 0.33, "Internal: GEA test set", color="#a8730a")
frame(ax, 4.04, ey - 0.03, 2.75, eh + 0.33, "External: ADEL", color="#9b3d74")
e1 = card(ax, 0.14, ey, 1.20, eh, "eval", "Classification", "AUROC with CIs;\nHolm-corrected\npaired tests")
e2 = card(ax, 1.40, ey, 1.20, eh, "eval", "Localisation", "false-positive tiles\nin IHC 0 cores\n(+1,326 OOF cores)")
e3 = card(ax, 2.66, ey, 1.20, eh, "eval", "Membrane, ISH", "student features;\n3+ vs 2+, FISH\nwithin 2+; routing")
e4 = card(ax, 4.12, ey, 1.25, eh, "ext", "Frozen models", "3 pixel sizes x\n2 stain settings;\nno retraining")
e5 = card(ax, 5.47, ey, 1.25, eh, "ext", "Site adaptation", "site IHC scores;\n5-fold CV x 3;\nprotocol fixed first")
path_arrow(ax, [(e4["r"], e4["cy"]), (e5["l"], e5["cy"])])
top = ey - 0.03 + eh + 0.33
bus = 1.66
path_arrow(ax, [(c3["cx"], ry - 0.09), (c3["cx"], bus), (3.40, bus), (3.40, top)])
path_arrow(ax, [(c3["cx"], bus), (5.42, bus), (5.42, top)])
label(ax, 4.62, bus + 0.17, "fold models, frozen", fontsize=8)
path_arrow(ax, [(6.70, d3["b"]), (6.70, 1.58), (6.20, 1.58), (6.20, top)], color="#9b3d74")
save(fig, "Fig1")

# ============================================================================ Fig. 2  SANC-MIL architecture
H2 = 6.30
fig, ax = canvas(H2)
def head(y, letter, text):
    ax.text(0.04, y, letter, fontsize=11, fontweight="bold", color=INK, va="top")
    ax.text(0.27, y - 0.02, text, fontsize=9, fontweight="bold", color=INK, va="top")
head(H2 - 0.03, "a", "Input: an IHC-stained core is a bag of tissue tiles; each tile gets two descriptions")
import os
if not os.path.exists(CORE):
    raise SystemExit("Fig. 2a needs an example IHC core image at figures/assets/example_core.jpg (see README).")
img = Image.open(CORE).convert("RGB"); S = img.size[0]; arr = np.asarray(img).astype(float); tis = arr.mean(-1) < 215
cx0, cy0, d = 0.92, 5.24, 1.40
iax = fig.add_axes([(cx0 - d / 2) / W, (cy0 - d / 2) / H2, d / W, d / H2]); iax.imshow(img); iax.axis("off")
n = 9; step = S / n
for i in range(n):
    for j in range(n):
        if tis[int(j * step):int((j + 1) * step), int(i * step):int((i + 1) * step)].mean() >= 0.5:
            iax.add_patch(Rectangle((i * step, j * step), step, step, fill=False, ec="#2563a8", lw=0.5))
label(ax, cx0, 4.24, "tiles with \u2265 50% tissue\n224 x 224 px at 0.46 \u00b5m/px\n(grid schematic, not to scale)")
tx = 1.86
for k, (fx, fy) in enumerate([(0.62, 0.30), (0.40, 0.58), (0.70, 0.62)]):
    c = img.crop((int(fx * S), int(fy * S), int(fx * S + step), int(fy * S + step)))
    tax = fig.add_axes([(tx + 0.09 * k) / W, (5.32 - 0.22 * k) / H2, 0.50 / W, 0.50 / H2]); tax.imshow(c); tax.set_xticks([]); tax.set_yticks([])
    for sp in tax.spines.values(): sp.set_edgecolor("#2563a8"); sp.set_linewidth(0.8)
path_arrow(ax, [(1.70, 5.12), (1.86, 5.12)])
label(ax, 2.22, 4.62, "tile $t = 1, \\ldots, n$")
b1 = card(ax, 2.94, 5.30, 1.84, 0.70, "fixed", "Frozen ResNet-50", "ImageNet weights, stages 1-3,\nglobal average pooling")
b2 = card(ax, 2.94, 4.20, 1.84, 0.84, "fixed", "Stain calibration (fixed)", "14 measures: DAB levels,\nmembranous and nuclear\nDAB, tissue, focus")
e1 = card(ax, 5.02, 5.42, 1.76, 0.46, "data", None, "$\\mathbf{e}_t \\in \\mathbb{R}^{1024}$  (embedding)")
e2 = card(ax, 5.02, 4.39, 1.76, 0.46, "data", None, "$\\tilde{\\mathbf{h}}_t \\in \\mathbb{R}^{14}$  (stain features)")
path_arrow(ax, [(2.66, 5.12), (2.80, 5.12), (2.80, b1["cy"]), (b1["l"], b1["cy"])]); path_arrow(ax, [(2.80, 5.12), (2.80, b2["cy"]), (b2["l"], b2["cy"])])
path_arrow(ax, [(b1["r"], b1["cy"]), (e1["l"], e1["cy"])]); path_arrow(ax, [(b2["r"], b2["cy"]), (e2["l"], e2["cy"])])
ax.plot([0.04, W - 0.04], [3.98, 3.98], color="#d0d3d8", lw=0.6)

head(3.93, "b", "SANC-MIL head (trained, 346,181 parameters): two scores per tile, additive pooling")
i1 = card(ax, 0.06, 3.02, 1.10, 0.46, "data", None, "$\\tilde{\\mathbf{h}}_t$ (14)")
i2 = card(ax, 0.06, 2.20, 1.10, 0.46, "data", None, "$\\mathbf{e}_t$ (1024)")
sp_ = card(ax, 1.30, 2.96, 1.12, 0.58, "train", "Stain projection", "Linear 14\u219264, ReLU")
fu = card(ax, 1.30, 1.98, 1.12, 0.84, "train", "Fuse, project", "concat (1,088)\nLinear\u2192256, ReLU,\ndropout 0.25: $\\mathbf{x}_t$")
path_arrow(ax, [(i1["r"], i1["cy"]), (sp_["l"], i1["cy"])]); path_arrow(ax, [(sp_["cx"], sp_["b"]), (sp_["cx"], fu["t"])])
path_arrow(ax, [(i2["r"], i2["cy"]), (fu["l"], i2["cy"])])
at = card(ax, 2.68, 2.78, 1.86, 0.92, "train", "Gated attention", "$\\alpha_t=\\mathbf{w}^{\\top}[\\tanh(V\\mathbf{x}_t)\\odot\\sigma(U\\mathbf{x}_t)]$\n$V, U$: 256\u2192128;  $\\mathbf{w}$: 128\u21921\nsoftmax over tiles \u2192 $a_t \\geq 0$")
ih = card(ax, 2.68, 1.82, 1.86, 0.72, "train", "Instance head", "dropout, Linear 256\u21921\nsigned tile logit $s_t$")
path_arrow(ax, [(fu["r"], fu["cy"]), (2.55, fu["cy"]), (2.55, at["cy"]), (at["l"], at["cy"])]); path_arrow(ax, [(2.55, fu["cy"]), (2.55, ih["cy"]), (ih["l"], ih["cy"])])
od = card(ax, 4.80, 3.16, 1.98, 0.58, "train", "Ordinal head (CORAL)", "$\\mathbf{u}=\\Sigma_t\\,a_t\\mathbf{x}_t$; Linear 256\u21921 + 3 biases")
oo = card(ax, 4.80, 2.66, 1.98, 0.32, "out", None, "$P(\\mathrm{IHC}>k)$, $k=0,1,2$; expected score")
bg = card(ax, 4.80, 1.86, 1.98, 0.62, "out", "Bag decision", "$z=\\Sigma_t\\,a_t s_t$;   $P(\\mathrm{HER2+})=\\sigma(z/T)$")
tm = card(ax, 4.80, 1.04, 1.98, 0.62, "out", "Tile positivity map", "$\\sigma(s_t)$ for every tile")
path_arrow(ax, [(od["cx"], od["b"]), (od["cx"], oo["t"])])
path_arrow(ax, [(at["r"], 3.45), (od["l"], 3.45)])
path_arrow(ax, [(at["r"], at["b"] + 0.16), (4.66, at["b"] + 0.16), (4.66, bg["cy"] + 0.10), (bg["l"], bg["cy"] + 0.10)])
path_arrow(ax, [(ih["r"], ih["cy"] + 0.14), (4.62 - 0.0, ih["cy"] + 0.14), (4.62, bg["cy"] - 0.12), (bg["l"], bg["cy"] - 0.12)])
path_arrow(ax, [(ih["r"], ih["cy"] - 0.18), (4.70, ih["cy"] - 0.18), (4.70, tm["cy"]), (tm["l"], tm["cy"])])
label(ax, 3.61, 1.40, "$a_t$: how much a tile matters (no sign)\n$s_t$: whether the tile is positive (signed)", fontsize=8, style="italic")
ax.plot([0.04, W - 0.04], [0.86, 0.86], color="#d0d3d8", lw=0.6)

head(0.81, "c", "Training losses: $\\mathcal{L}=\\mathcal{L}_{status}+0.5\\,\\mathcal{L}_{ord}+0.5\\,\\mathcal{L}_{neg}+0.5\\,\\mathcal{L}_{top\\text{-}k}$")
lw_ = (W - 0.12 - 3 * 0.08) / 4
for k, (t, b) in enumerate([("$\\mathcal{L}_{status}$", "BCE on $z$; every bag"), ("$\\mathcal{L}_{ord}$", "cumulative BCE; every bag"),
                            ("$\\mathcal{L}_{neg}$", "each $s_t$, IHC 0/1+ bags \u2192 0"), ("$\\mathcal{L}_{top\\text{-}k}$", "top-5% mean $s_t$, HER2+ \u2192 1")]):
    card(ax, 0.06 + k * (lw_ + 0.08), 0.06, lw_, 0.50, "loss", t, b, hb=0.22, dashed=True)
save(fig, "Fig2")

# ============================================================================ Fig. 3  supervision by label
fig, ax = canvas(2.36)
rng = np.random.default_rng(7)
cols = [("IHC 0 or 1+", "HER2-negative by definition", "neg", "data"), ("IHC 2+, FISH-negative", "equivocal staining", "eq", "fixed"),
        ("IHC 3+ or 2+, FISH-positive", "HER2-positive", "pos", "loss")]
cw = (W - 0.10 - 2 * 0.08) / 3
for k, (t1, t2, kind, pk) in enumerate(cols):
    x0 = 0.05 + k * (cw + 0.08)
    card(ax, x0, 0.05, cw, 2.26, pk, t1, None, hb=0.24)
    ax.text(x0 + cw / 2, 1.93, t2, ha="center", va="center", fontsize=8, color=GREY)
    ccx, ccy, r = x0 + cw / 2, 1.30, 0.46
    ax.add_patch(Circle((ccx, ccy), r, fc="#f7f4ec", ec="#9aa0a6", lw=0.8, zorder=3))
    g = 0.145; hot = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            if (i * g) ** 2 + (j * g) ** 2 > (r - 0.08) ** 2: continue
            xx, yy = ccx + i * g - 0.064, ccy + j * g - 0.064
            txt = ""
            if kind == "neg": fc, txt = "#cfdcef", "\u2193"
            elif kind == "eq": fc = "#e6e4de"
            else:
                fc = "#e6e4de"
                if rng.random() < 0.3 and hot < 3: fc, txt, hot = "#e2795a", "\u2191", hot + 1
            ax.add_patch(Rectangle((xx, yy), 0.128, 0.128, fc=fc, ec="white", lw=0.6, zorder=4))
            if txt: ax.text(xx + 0.064, yy + 0.062, txt, ha="center", va="center", fontsize=8, color=INK, zorder=5)
    body = {"neg": "every tile is a known negative\n$\\mathcal{L}_{status}$ ($y=0$) and $\\mathcal{L}_{ord}$\n$\\mathcal{L}_{neg}$ pushes every $s_t$ down",
            "eq": "weak membranous staining is present,\nso no tile is assumed negative\n$\\mathcal{L}_{status}$ ($y=0$) and $\\mathcal{L}_{ord}$ only",
            "pos": "some tiles are positive,\nbut which ones is unknown\n$\\mathcal{L}_{status}$ ($y=1$) and $\\mathcal{L}_{ord}$\n$\\mathcal{L}_{top\\text{-}k}$ pushes the top 5% of $s_t$ up"}[kind]
    ax.text(x0 + cw / 2, 0.76, body, ha="center", va="top", fontsize=8, color=INK, linespacing=1.3)
save(fig, "Fig3")
print("ok")

# ============================================================================ Fig. S6  membrane quantification pipeline
fig, ax = canvas(3.30)
frame(ax, 0.06, 1.78, W - 0.12, 1.42, "Training cores only", color="#16805a")
frame(ax, 0.06, 0.06, W - 0.12, 1.42, "Test cores (and ADEL slides, unchanged)", color="#a8730a", lx=1.55)
y1, h1 = 1.90, 1.12
r1 = card(ax, 0.16, y1, 1.42, h1, "fixed", "Rule detector", "colour deconvolution\nridge + top-hat test\nlabel-anchored\nDAB bands")
r2 = card(ax, 1.76, y1 + 0.40, 1.50, h1 - 0.40, "data", "Pseudo-labels", "12,432 crops (512 px)\nignore + confidence")
r3 = card(ax, 1.76, y1, 1.50, 0.32, "data", None, "OOF SANC-MIL tile scores")
r4 = card(ax, 3.44, y1, 1.86, h1, "train", "Segmentation student", "ResNet-34 encoder, gated\nskips, transformer bottleneck\nconfidence-weighted\nCE + soft Dice")
r5 = card(ax, 5.46, y1, 1.24, h1, "fixed", "Compared with", "plain U-Net\nnaive hard labels\nignore only\n(same budget)", dashed=True)
path_arrow(ax, [(r1["r"], r2["cy"]), (r2["l"], r2["cy"])])
path_arrow(ax, [(r2["r"], r2["cy"]), (r4["l"], r2["cy"])]); path_arrow(ax, [(r3["r"], r3["cy"]), (r4["l"], r3["cy"])])
path_arrow(ax, [(r4["r"], r4["cy"]), (r5["l"], r4["cy"])], dashed=True, head=False)
y2, h2 = 0.16, 1.12
t1 = card(ax, 0.16, y2, 1.30, h2, "train", "Whole-core masks", "student inference\n4-fold TTA\n2.6 s per core")
t2 = card(ax, 1.62, y2, 1.40, h2, "fixed", "Cell coverage", "watershed seeded\non nuclei; share of\nboundary on\nmembranous DAB")
t3 = card(ax, 3.18, y2, 1.62, h2, "fixed", "Core features", "12 interior crops;\nmembrane area per band;\ncells with ≥ 30, 50,\n70% coverage")
t4 = card(ax, 4.96, y2, 1.74, h2, "eval", "Analyses", "trend with IHC score;\n3+ vs 2+; FISH within 2+;\ndirection fixed on training\ncores; Holm correction")
for a_, b_ in [(t1, t2), (t2, t3), (t3, t4)]: path_arrow(ax, [(a_["r"], a_["cy"]), (b_["l"], b_["cy"])])
path_arrow(ax, [(r4["cx"] - 0.30, y1), (r4["cx"] - 0.30, 1.62), (t1["cx"], 1.62), (t1["cx"], y2 + h2)])
label(ax, 2.10, 1.62, "trained student, frozen", fontsize=8)
save(fig, "FigS6")
print("S6 ok")
