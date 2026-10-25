"""Shared print-size style for MBEC: Liberation Sans (Arial metrics), every text >= 8 pt, lines >= 0.5 pt."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
MIN_PT = 8.0
OUTDIR = _os.path.join(ROOT, "figures", "output") + "/"
STYLE = {"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600}
def enforce(fig):
    n = 0
    for t in fig.findobj(Text):
        if t.get_text() and t.get_fontsize() < MIN_PT: t.set_fontsize(MIN_PT); n += 1
    return n
def report(fig):
    sizes = sorted({round(t.get_fontsize(), 1) for t in fig.findobj(Text) if t.get_text()})
    w, h = fig.get_size_inches(); return f"{w*25.4:.0f} x {h*25.4:.0f} mm, font sizes {sizes}"
