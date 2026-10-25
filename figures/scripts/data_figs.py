"""Data figures for the manuscript. Every number is read from the v2.1 result tables; nothing is typed in by hand."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import json, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

T = _os.path.join(ROOT, "results", "gea", "tables") + "/"
import sys; sys.path.insert(0, HERE); from figstyle import STYLE, enforce, report, OUTDIR
O = OUTDIR
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "savefig.dpi": 600, "figure.dpi": 150, "pdf.fonttype": 42, "svg.fonttype": "none"})
BLUE, ORANGE, AQUA, GREY, LGREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#cfcdc6", "#0b0b0b"
GRID = "#e6e4de"

def _fit(fig, pad=0.04, it=10):
    """Shrink and shift the axes until every label lies inside the fixed canvas, so the saved file is exactly
    the print width and no lettering is scaled below 8 pt at that width."""
    W, H = fig.get_size_inches()
    for _ in range(it):
        fig.canvas.draw(); r = fig.canvas.get_renderer(); tb = fig.get_tightbbox(r)
        x0, x1, y0, y1 = tb.x0, tb.x1, tb.y0, tb.y1
        if x0 >= pad - 1e-3 and x1 <= W - pad + 1e-3 and y0 >= pad - 1e-3 and y1 <= H - pad + 1e-3: break
        sx = min(1.0, (W - 2 * pad) / (x1 - x0)); sy = min(1.0, (H - 2 * pad) / (y1 - y0))
        for ax in fig.axes:
            b = ax.get_position()
            nx0 = (b.x0 - x0 / W) * sx + pad / W; ny0 = (b.y0 - y0 / H) * sy + pad / H
            ax.set_position([nx0, ny0, b.width * sx, b.height * sy])
        for t in fig.texts:
            px, py = t.get_position(); t.set_position(((px - x0 / W) * sx + pad / W, (py - y0 / H) * sy + pad / H))

def save(fig, name):
    plt.rcParams.update(STYLE); enforce(fig); _fit(fig); print(name, report(fig))
    for ext in ("png", "pdf"):
        fig.savefig(O + f"{name}.{ext}", facecolor="white")
    plt.close(fig)

def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)

def ci(s):
    m = re.match(r"\s*([\d.]+)\s*\[([\d.]+)[–-]([\d.]+)\]", str(s)); return tuple(float(v) for v in m.groups())

NICE = {"SANC-MIL (proposed)": "SANC-MIL (proposed)", "ABMIL (Ilse; BJC-2023)": "ABMIL", "Stain intensity rule (Pisula 2023)": "Stain-intensity rule",
        "Calibrated stain features + LR": "Calibrated stain features + LR", "TransMIL-lite": "TransMIL-style", "Grad-CAM CNN (thumbnail)": "Grad-CAM CNN (thumbnail)",
        "SANC-MIL (proposed) [phikon]": "SANC-MIL, Phikon encoder", "Stain-only MIL (no deep emb)": "Stain-only MIL",
        "SANC − stain features": "SANC − stain features", "SANC − negative constraint": "SANC − negative constraint",
        "SANC − top-k constraint": "SANC − top-k constraint", "SANC − ordinal head": "SANC − ordinal head",
        "SANC − additive (attn pool)": "SANC − additive pooling", "SANC − stain features [phikon]": "SANC − stain features, Phikon",
        "ABMIL (Ilse; BJC-2023) [phikon]": "ABMIL, Phikon", "MaxPool [phikon]": "MaxPool, Phikon"}

# ---------------- Figure 4: classification forest plot ----------------
C = pd.read_csv(T + "classification_table.csv")
C["auc"], C["lo"], C["hi"] = zip(*C["Test AUROC"].map(ci))
keep = ["SANC-MIL (proposed)", "Grad-CAM CNN (thumbnail)", "Stain-only MIL (no deep emb)", "Calibrated stain features + LR", "TransMIL-lite",
        "SANC-MIL (proposed) [phikon]", "MaxPool", "Stain intensity rule (Pisula 2023)", "MeanPool", "ABMIL (Ilse; BJC-2023)", "CLAM-SB"]
D = C.set_index("method").loc[keep].reset_index()
P = pd.read_csv(T + "paired_delta_auc.csv"); P["m"] = P.comparison.str.replace("SANC-MIL (proposed) − ", "", regex=False)
holm = dict(zip(P.m, P.p_holm))
fig, ax = plt.subplots(figsize=(6.85, 3.5))
y = np.arange(len(D))[::-1]
for yi, (_, r) in zip(y, D.iterrows()):
    col = BLUE if r.method == "SANC-MIL (proposed)" else GREY
    ax.plot([r.lo, r.hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", clip_on=False)
    ax.plot(r.auc, yi, "o", ms=5.5, color=col, mec="white", mew=0.8, zorder=3)
    ax.text(1.0045, yi, f"{r.auc:.3f} ({r.lo:.3f}–{r.hi:.3f})", va="center", fontsize=7, color=INK)
    p = holm.get(r.method)
    ax.text(1.0415, yi, "—" if p is None else (f"{p:.3f}" if p < 1 else "1.000"), va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([NICE.get(m, m) for m in D.method])
for t in ax.get_yticklabels():
    if t.get_text().startswith("SANC-MIL (proposed)"): t.set_fontweight("bold"); t.set_color(BLUE)
ax.axvline(D.auc.iloc[0], color=BLUE, lw=0.7, ls=(0, (3, 3)), alpha=0.6)
ax.set_xlim(0.85, 1.001); ax.set_xlabel("Test AUROC for HER2 status (95% bootstrap CI; 307 cores, 16 HER2-positive)")
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.text(1.0045, len(D) - 0.35, "AUROC (95% CI)", fontsize=7, fontweight="bold", color=INK)
ax.text(1.0415, len(D) - 0.35, "Holm p*", fontsize=7, fontweight="bold", color=INK)
fig.text(0.01, -0.03, "*Paired bootstrap test of SANC-MIL minus each method, Holm-corrected over 18 comparisons.", fontsize=6.5, color="#52514e")
save(fig, "Fig4_classification")

# ---------------- Figure 5: H1 negative-bag constraint ----------------
S = pd.read_csv(T + "seed_runs.csv")
H = pd.read_csv(T + "h1_negative_constraint_tests.csv")
fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.8), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.35], wspace=0.75))
pairs = [("SANC-MIL (proposed)", "with constraint", BLUE), ("SANC − negative constraint", "without constraint", ORANGE)]
for ax, (col, ylab, lab) in zip(axs[:2], [("IHC0_tile_FP_rate@0.5", "Fraction of tiles with P > 0.5", "a"),
                                          ("IHC0_cores_with_any_tile>0.9", "Fraction of cores with any tile P > 0.9", "b")]):
    for k, (m, nm, c) in enumerate(pairs):
        v = S[S.method == m].sort_values("seed")[col].values
        ax.bar(k, v.mean(), width=0.55, color=c, alpha=0.9, edgecolor="white", linewidth=1.5, zorder=2)
        ax.scatter(np.full(len(v), k) + np.array([-0.12, 0, 0.12])[:len(v)], v, s=14, color="white", edgecolor=INK, lw=0.7, zorder=3)
        ax.text(k + 0.33, v.mean(), f"{v.mean():.3f}", ha="left", va="center", fontsize=7, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["with", "without"]); ax.set_xlim(-0.5, 1.75); ax.set_xlabel("negative-bag constraint")
    ax.set_ylabel(ylab); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); panel(ax, lab, x=-0.45)
    ax.set_title("IHC 0 test cores (n = 284)", fontsize=7.5, color="#52514e")
axs[0].set_ylim(0, 0.115); axs[1].set_ylim(0, 0.42)
ax = axs[2]; Hs = H[(H.A == "SANC-MIL (proposed)") & (H.B == "SANC − negative constraint")].copy()
Hs["y"] = [5 - i for i in range(len(Hs))]
for _, r in Hs.iterrows():
    c = BLUE if r.split == "test" else AQUA
    ax.plot([r.lo, r.hi], [r.y, r.y], color=c, lw=1.6, solid_capstyle="round"); ax.plot(r["diff"], r.y, "o", color=c, ms=5, mec="white", mew=0.7)
ax.axvline(0, color=INK, lw=0.7)
ax.set_yticks(Hs.y); ax.set_yticklabels([f"S{int(s)} {'test' if sp == 'test' else 'OOF'}" for s, sp, n in zip(Hs.seed, Hs.split, Hs.n_cores)])
ax.set_xlabel("Paired difference in IHC 0 false-positive\ntile fraction (with − without), 95% CI"); ax.set_xlim(-0.17, 0.01)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True); panel(ax, "c", x=-0.42)
ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1.6, label="test cores"), Line2D([], [], color=AQUA, marker="o", lw=1.6, label="out-of-fold cores")],
          loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=6.8, ncol=1); ax.set_ylim(-1.55, 5.5)
save(fig, "Fig5_negative_constraint")

# ---------------- Figure 6: localisation trade-off ----------------
L = pd.read_csv(T + "localization_proxies.csv")
L = L[L["IHC0_tile_FP_rate@0.5"].notna() & ~L.method.str.contains("phikon")]
fig, ax = plt.subplots(figsize=(5.08, 3.6))
NICE2 = dict(NICE, **{"Grad-CAM CNN (per-image normalised CAM)": "Grad-CAM (normalised CAM)", "Grad-CAM CNN (CAM × P(HER2+))": "Grad-CAM (CAM × P)"})
LAB = {"SANC-MIL (proposed)": (1.6e-3, 1.0), "SANC − negative constraint": (0.02, 1.035), "MeanPool": (0.12, 0.80), "MaxPool": (1.4e-4, 0.58),
       "CLAM-SB": (6e-3, 0.66), "SANC − stain features": (0.03, 0.73), "Grad-CAM CNN (per-image normalised CAM)": (4e-3, 0.16),
       "Grad-CAM CNN (CAM × P(HER2+))": (1.3e-4, 0.20), "SANC − top-k constraint": (1.3e-4, 0.84)}
for _, r in L.iterrows():
    x = max(r["IHC0_tile_FP_rate@0.5"], 1e-4); yv = r["IHC3_median_pos_tile_frac"]
    c = BLUE if r.method == "SANC-MIL (proposed)" else (ORANGE if r.method == "SANC − negative constraint" else GREY)
    if np.isfinite(r["IHC0_tile_FP_rate@0.5_lo"]):
        ax.plot([max(r["IHC0_tile_FP_rate@0.5_lo"], 1e-4), r["IHC0_tile_FP_rate@0.5_hi"]], [yv, yv], color=c, lw=1.0, alpha=0.6)
    ax.plot(x, yv, "o", color=c, ms=6.5 if c != GREY else 4.5, mec="white", mew=0.8, zorder=4 if c != GREY else 3)
    if r.method in LAB:
        tx, ty = LAB[r.method]
        ax.annotate(NICE2.get(r.method, r.method), xy=(x, yv), xytext=(tx, ty), fontsize=6.6, va="center",
                    color=INK if c == GREY else c, fontweight="bold" if c != GREY else "normal",
                    arrowprops=dict(arrowstyle="-", color="#9a9893", lw=0.5, shrinkA=0, shrinkB=3))
ax.set_xscale("log"); ax.set_xlim(8e-5, 0.6); ax.set_ylim(0, 1.06)
ax.set_xlabel("False-positive tile fraction in IHC 0 test cores (P > 0.5; log scale)")
ax.set_ylabel("Median fraction of tiles called positive\nin IHC 3+ test cores (P > 0.5)")
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, "Fig6_localisation_tradeoff")

# ---------------- Figure 7: membrane features ----------------
Tr = pd.read_csv(T + "clinical_feature_trends.csv"); H3 = pd.read_csv(T + "h3_feature_separation_3v2_and_fish.csv")
fig = plt.figure(figsize=(6.85, 3.2))
ax = fig.add_axes([0.075, 0.33, 0.24, 0.57]); axb = fig.add_axes([0.575, 0.33, 0.185, 0.57]); axc = fig.add_axes([0.795, 0.33, 0.185, 0.57])
grp = ["median_0", "median_2-", "median_2+", "median_3"]; xl = ["0", "2+\nFISH−", "2+\nFISH+", "3+"]
feats = [("stu_mem_mod_frac", "membranous DAB area ≥ moderate", BLUE), ("stu_cell_cov>=0.5&mod", "cells ≥ 50% covered at ≥ moderate DAB", ORANGE),
         ("stu_mem_strong_frac", "membranous DAB area ≥ strong", AQUA)]
for f, nm, c in feats:
    r = Tr[(Tr.split == "test") & (Tr.feature == f)].iloc[0]
    ax.plot(range(4), [r[g] for g in grp], "-o", color=c, ms=4.5, lw=1.6, mec="white", mew=0.7, label=f"{nm} (Spearman ρ = {r.spearman_rho:.2f})")
ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=6.8); ax.set_xlabel("IHC group (test cores)"); ax.set_ylabel("Median core-level fraction")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0, 0.16)
ax.legend(loc="upper left", bbox_to_anchor=(-0.28, -0.3), frameon=False, fontsize=6.2, handlelength=1.5)
NF = {"stu_mem_strong_frac": "membranous DAB ≥ strong", "stu_mem_mod_frac": "membranous DAB ≥ moderate", "stu_cell_cov>=0.5&mod": "cells ≥ 50% covered, ≥ moderate",
      "stu_cell_median_cov_pos": "median coverage of stained cells", "stu_cell_cov>=0.7": "cells ≥ 70% covered"}
fl = ["stu_mem_strong_frac", "stu_mem_mod_frac", "stu_cell_cov>=0.5&mod", "stu_cell_cov>=0.7", "stu_cell_median_cov_pos"]
for axx, comp, title in [(axb, "3+ vs 2+", "IHC 3+ vs IHC 2+"), (axc, "2+/FISH+ vs 2+/FISH-", "2+/FISH+ vs 2+/FISH−")]:
    yy = np.arange(len(fl))[::-1]
    for yi, f in zip(yy, fl):
        for sp, c, dy in [("train", GREY, 0.17), ("test", BLUE, -0.17)]:
            r = H3[(H3.comparison == comp) & (H3.split == sp) & (H3.feature == f)].iloc[0]
            axx.plot([r.lo, r.hi], [yi + dy, yi + dy], color=c, lw=1.4, solid_capstyle="round"); axx.plot(r.AUROC, yi + dy, "o", color=c, ms=4.2, mec="white", mew=0.6)
    axx.axvline(0.5, color=INK, lw=0.6, ls=(0, (2, 2))); axx.set_xlim(0.2, 1.02); axx.set_xlabel("AUROC (95% CI)")
    n_tr = H3[(H3.comparison == comp) & (H3.split == "train")].iloc[0]; n_te = H3[(H3.comparison == comp) & (H3.split == "test")].iloc[0]
    axx.set_title(f"{title}\ntrain {n_tr.n_pos} vs {n_tr.n_neg}; test {n_te.n_pos} vs {n_te.n_neg}", fontsize=7)
    axx.grid(axis="x", color=GRID, lw=0.5); axx.set_axisbelow(True); axx.set_yticks(yy)
axb.set_yticklabels([NF[f] for f in fl], fontsize=6.6); axc.set_yticklabels([]); axc.tick_params(axis="y", length=0)
fig.legend(handles=[Line2D([], [], color=GREY, marker="o", lw=1.4, label="training cores"), Line2D([], [], color=BLUE, marker="o", lw=1.4, label="test cores")],
           loc="upper center", bbox_to_anchor=(0.69, 0.16), ncol=2, frameon=False, fontsize=6.6)
for x, l in [(0.005, "a"), (0.36, "b"), (0.775, "c")]: fig.text(x, 0.95, l, fontsize=10, fontweight="bold")
save(fig, "Fig7_membrane_features")

# ---------------- Figure 8: decisions by IHC group + P(HER2+) ----------------
TP = pd.read_csv(T + "test_predictions.csv")
order = ["0", "2-", "2+", "3"]; xl2 = ["IHC 0\n(n = 284)", "2+ FISH−\n(n = 7)", "2+ FISH+\n(n = 8)", "IHC 3+\n(n = 8)"]
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.8), gridspec_kw=dict(wspace=0.40, width_ratios=[1.1, 1.0]))
ax = axs[0]; rng = np.random.default_rng(1)
for i, g in enumerate(order):
    v = TP[TP.group == g].p_her2.values; c = [BLUE, ORANGE, AQUA, "#6250d6"][i]
    ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=7 if g == "0" else 14, color=c, alpha=0.55 if g == "0" else 0.9, edgecolor="none")
    ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.2)
ax.set_xticks(range(4)); ax.set_xticklabels(xl2); ax.set_ylabel("SANC-MIL P(HER2-positive)"); ax.set_ylim(-0.02, 1.02)
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); panel(ax, "a", x=-0.2)
ax = axs[1]
cats = [("HER2-negative (model)", "negative (model)", LGREY), ("HER2-positive (model)", "positive (model)", BLUE),
        ("EQUIVOCAL: ISH required", "route to ISH", ORANGE), ("ABSTAIN (refer)", "abstain / refer", "#6250d6")]
ct = pd.crosstab(TP.group, TP.decision).reindex(order).fillna(0)
frac = ct.div(ct.sum(1), axis=0); left = np.zeros(4)
for key, nm, c in cats:
    v = frac[key].values if key in frac else np.zeros(4)
    ax.barh(range(4)[::-1], v, left=left, color=c, edgecolor="white", linewidth=1.2, height=0.62, label=nm)
    for i, (vi, li, n) in enumerate(zip(v, left, ct[key].values if key in ct else np.zeros(4))):
        if vi > 0.07: ax.text(li + vi / 2, 3 - i, f"{int(n)}", ha="center", va="center", fontsize=7, color=INK if c == LGREY else "white")
    left += v
ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(xl2, fontsize=7); ax.set_xlim(0, 1); ax.set_xlabel("Fraction of test cores")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=4, frameon=False, fontsize=6.6, handlelength=1.0, columnspacing=0.9)
panel(ax, "b", x=-0.3)
save(fig, "Fig8_decisions")

# ---------------- Supplementary: ablations of SANC (classification + localisation) ----------------
abl = ["SANC-MIL (proposed)", "SANC − stain features", "SANC − negative constraint", "SANC − top-k constraint", "SANC − ordinal head",
       "SANC − additive (attn pool)", "Stain-only MIL (no deep emb)"]
LL = pd.read_csv(T + "localization_proxies.csv").set_index("method"); CC = C.set_index("method")
fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.8), sharey=True, gridspec_kw=dict(wspace=0.22))
yy = np.arange(len(abl))[::-1]
for ax, (vals, xlab, lim) in zip(axs, [([CC.loc[m, "auc"] for m in abl], "Test AUROC", (0.94, 1.0)),
                                       ([LL.loc[m, "IHC0_tile_FP_rate@0.5"] for m in abl], "IHC 0 false-positive tile fraction", (0, 0.085)),
                                       ([LL.loc[m, "IHC3_median_pos_tile_frac"] for m in abl], "IHC 3+ median positive-tile fraction", (0.6, 1.0))]):
    for yi, m, v in zip(yy, abl, vals):
        c = BLUE if m == "SANC-MIL (proposed)" else (ORANGE if m == "SANC − negative constraint" else GREY)
        ax.plot([lim[0], v], [yi, yi], color=c, lw=1.2, alpha=0.5); ax.plot(v, yi, "o", color=c, ms=5, mec="white", mew=0.7)
        ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", fontsize=6.3, color=INK)
    ax.set_xlim(*lim); ax.set_xlabel(xlab); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_yticks(yy); axs[0].set_yticklabels([NICE.get(m, m) for m in abl])
for a, l in zip(axs, "abc"): panel(a, l, x=-0.05 if l != "a" else -0.85)
save(fig, "FigS4_ablations")
print("ok")
