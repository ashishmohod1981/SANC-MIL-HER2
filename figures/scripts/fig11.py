"""Figure 11: ADEL site adaptation. Numbers read from cell E2 output files."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open(_os.path.join(HERE, "data_figs.py")).read().split("# ---------------- Figure 4")[0])
R = _os.path.join(ROOT, "results", "adel_site_adaptation") + "/"
S = pd.read_csv(R + "external_adel_site_adaptation_summary.csv"); P = pd.read_csv(R + "external_adel_site_adaptation_per_slide.csv")
T = pd.read_csv(R + "external_adel_site_adaptation_thresholds.csv")
rng = np.random.default_rng(2024)
def bci(x, n=4000):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return x.mean(), *np.percentile(b, [2.5, 97.5])
COND = [("SANC-MIL | frozen", "SANC\nfrozen"), ("SANC-MIL | adapted", "SANC\nadapted"),
        ("without constraint | frozen", "No constr.\nfrozen"), ("without constraint | adapted", "No constr.\nadapted")]
COL = {"SANC-MIL": BLUE, "without constraint": GREY}
au = S[S.family == "status_AUROC"].set_index("measure")
out = {}
fig = plt.figure(figsize=(6.85, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.36, left=0.09, right=0.985, top=0.95, bottom=0.14)
# (a) AUROC
ax = fig.add_subplot(gs[0, 0]); x = np.arange(4)
for j, (q, mk, lab) in enumerate((("status logit", "o", "Status logit"), ("ordinal head", "s", "Ordinal head (expected score)"))):
    for i, (k, _) in enumerate(COND):
        r = au.loc[f"{k} | {q}"]; col = COL[k.split(" | ")[0]]
        ax.plot([i + (j - 0.5) * 0.28] * 2, [r.lo, r.hi], color=col, lw=1.5, solid_capstyle="round")
        ax.plot(i + (j - 0.5) * 0.28, r.value, mk, ms=5.5, color=col if j == 0 else "white", mec=col, mew=1.2, zorder=3)
ax.plot([], [], "o", color=INK, ms=5, label="Status logit"); ax.plot([], [], "s", color="white", mec=INK, ms=5, label="Ordinal head")
ax.set_xticks(x); ax.set_xticklabels([c[1] for c in COND], fontsize=6.8); ax.set_ylim(0.80, 1.0)
ax.set_ylabel("Status AUROC (IHC 0/1+ vs 3+)"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.8, loc="lower right"); ax.set_title("Status discrimination", loc="left"); panel(ax, "a", x=-0.2)
# (b) negatives
ax = fig.add_subplot(gs[0, 1]); w = 0.36
for j, (g, nm, hatch) in enumerate(((0, "IHC 0", None), (1, "IHC 1+", "////"))):
    for i, (k, _) in enumerate(COND):
        m, lo, hi = bci(P[P.ihc_score == g][f"fp::{k}"]); out[f"fp_{g}_{k}"] = (m, lo, hi)
        col = COL[k.split(" | ")[0]]
        ax.bar(i + (j - 0.5) * w, m, w * 0.92, color=col if j == 0 else "white", edgecolor=col, hatch=hatch, lw=0.8, zorder=2)
        ax.errorbar(i + (j - 0.5) * w, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=3)
from matplotlib.patches import Patch
LEG_B = [Patch(facecolor="#6b6a66", edgecolor="#6b6a66", label="IHC 0 (n = 138)"), Patch(facecolor="white", edgecolor="#6b6a66", hatch="////", label="IHC 1+ (n = 102)")]
ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in COND], fontsize=6.8); ax.set_ylim(0, 0.6)
ax.set_ylabel("Mean fraction of tiles called positive"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=LEG_B, frameon=False, fontsize=6.8, loc="upper left"); ax.set_title("False positives in HER2-negative slides", loc="left"); panel(ax, "b", x=-0.2)
# (c) 3+
ax = fig.add_subplot(gs[1, 0])
for i, (k, _) in enumerate(COND):
    m, lo, hi = bci(P[P.ihc_score == 3][f"fp::{k}"]); out[f"fp_3_{k}"] = (m, lo, hi); col = COL[k.split(" | ")[0]]
    ax.bar(i, m, 0.55, color=col, zorder=2); ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in COND], fontsize=6.8); ax.set_ylim(0, 1)
ax.set_ylabel("Mean fraction of tiles called positive"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("Positive coverage in IHC 3+ slides (n = 103)", loc="left"); panel(ax, "c", x=-0.2)
# (d) operating points
ax = fig.add_subplot(gs[1, 1]); g = T.groupby(["model", "weights", "threshold"])[["sensitivity", "specificity"]].mean().reset_index()
for _, r in g.iterrows():
    col = COL[r.model]; mk = "o" if r.weights == "adapted" else "^"; fill = col if r.threshold == "site threshold" else "white"
    ax.plot(1 - r.specificity, r.sensitivity, mk, ms=6.5, color=fill, mec=col, mew=1.3, zorder=3)
for m in ("SANC-MIL", "without constraint"):
    for wts in ("frozen", "adapted"):
        a = g[(g.model == m) & (g.weights == wts)].set_index("threshold")
        ax.annotate("", xy=(1 - a.loc["site threshold", "specificity"], a.loc["site threshold", "sensitivity"]),
                    xytext=(1 - a.loc["GEA threshold", "specificity"], a.loc["GEA threshold", "sensitivity"]),
                    arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=0.7, alpha=0.7, shrinkA=4, shrinkB=4))
ax.set_xlim(0, 0.5); ax.set_ylim(0.70, 1.01); ax.set_xlabel("1 − specificity (HER2-negative slides called positive)"); ax.set_ylabel("Sensitivity (3+ slides)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="^", ls="", mec=INK, mfc="white", label="frozen"), Line2D([], [], marker="o", ls="", mec=INK, mfc="white", label="adapted"),
     Line2D([], [], marker="s", ls="", mec=INK, mfc="white", label="GEA threshold (open)"), Line2D([], [], marker="s", ls="", mec=INK, mfc=INK, label="site threshold (filled)"),
     Line2D([], [], color=BLUE, label="SANC-MIL"), Line2D([], [], color=GREY, label="No constraint")]
ax.legend(handles=h, frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.9, handlelength=1.4)
ax.set_title("Operating points (mean of 3 repeats)", loc="left"); panel(ax, "d", x=-0.2)
save(fig, "Fig11_site_adaptation")
json.dump(out, open(_os.path.join(ROOT, "results", "derived", "fig11_ci.json"), "w"), indent=1)
for k, v in out.items(): print(k, np.round(v, 3))
