"""Figure 10 and Table 7 values for the ADEL external evaluation. All numbers are read from the per-slide result files."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__)); ROOT = _os.path.abspath(_os.path.join(HERE, "..", ".."))
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
exec(open(_os.path.join(HERE, "data_figs.py")).read().split("# ---------------- Figure 4")[0])
E3 = _os.path.join(ROOT, "results", "adel_external") + "/"
REC = set(open(_os.path.join(ROOT, "results", "derived", "recovered.txt")).read().split())
A, B = "SANC-MIL", "SANC without negative constraint"
rng = np.random.default_rng(2024)
def load(m):
    d = pd.read_csv(f"{E3}mpp{m}/external_adel_mpp{m}_per_item.csv"); d["slide"] = d.item_id.str.extract(r"^(HER2_\d+)")[0]
    d["rec"] = d.slide.isin(REC); return d
def bci(x, n=4000):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return x.mean(), *np.percentile(b, [2.5, 97.5])
def paired(z):
    a, b = z[f"tileFP::{A}"].values, z[f"tileFP::{B}"].values; m, lo, hi = bci(a - b)
    return dict(n=len(z), A=a.mean(), B=b.mean(), d=m, lo=lo, hi=hi, p=wilcoxon(a, b).pvalue, better=int((a < b).sum()), worse=int((a > b).sum()),
                gt9A=float((z[f"tileMax::{A}"] > 0.9).mean()), gt9B=float((z[f"tileMax::{B}"] > 0.9).mean()))
S = {}
for m in ["022", "025", "028"]:
    d = load(m); summ = pd.read_csv(f"{E3}mpp{m}/external_adel_mpp{m}_summary.csv")
    for mode in ["frozen", "refit"]:
        x = d[d["mode"] == mode]; r = {}
        r["h1"] = paired(x[x.ihc_score == 0]); r["h1_norec"] = paired(x[(x.ihc_score == 0) & ~x.rec]); r["h1_1plus"] = paired(x[x.ihc_score == 1])
        s = summ[summ["mode"] == mode].set_index("method")
        r["auc"] = {k: (s.loc[k, "status_AUROC"], s.loc[k, "AUROC_lo"], s.loc[k, "AUROC_hi"]) for k in [A, B, "ABMIL", "Stain-intensity rule", "Membrane + cell features"]}
        st = x[(x.ihc_score != 2) & ~x.rec]; y = (st.ihc_score == 3).astype(int)
        r["auc_norec"] = {k: roc_auc_score(y, st[f"logit::{k}"]) for k in [A, B]}
        r["rho_exp"] = {k: s.loc[f"{k} (expected ordinal score)", "spearman_vs_score"] for k in [A, B]}
        r["qwk"] = {k: s.loc[f"{k} (expected ordinal score)", "QWK_vs_score"] for k in [A, B]}
        r["rho_strong"] = s.loc["feature: membranous DAB area, ≥strong (student)", "spearman_vs_score"]
        r["rho_cov7"] = s.loc["feature: cells ≥70% membrane-covered (student)", "spearman_vs_score"]
        r["posfrac"] = {k: {int(g): bci(v[f"tileFP::{k}"]) for g, v in x.groupby("ihc_score")} for k in [A, B]}
        r["strong"] = {int(g): v["stu_mem_strong_frac"].values.tolist() for g, v in x.groupby("ihc_score")}
        S[f"{m}_{mode}"] = r
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "strong"} for k, v in S.items()},
          open(_os.path.join(ROOT, "results", "derived", "table7.json"), "w"), indent=1, default=float)

# ---------------- Figure 10 ----------------
fig = plt.figure(figsize=(6.85, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.42, left=0.09, right=0.985, top=0.95, bottom=0.16)
SC = [("022", "0.22"), ("025", "0.25"), ("028", "0.28")]
# (a) H1 forest
ax = fig.add_subplot(gs[0, 0]); rows = []
for m, lab in SC:
    for mode in ["frozen", "refit"]:
        rows.append((f"{lab} µm/px, {'frozen' if mode == 'frozen' else 're-estimated'} stain", S[f"{m}_{mode}"]["h1"], m == "025" and mode == "frozen"))
yy = np.arange(len(rows))[::-1]
for yi, (lab, h, prim) in zip(yy, rows):
    col = BLUE if prim else GREY
    ax.plot([h["lo"], h["hi"]], [yi, yi], color=col, lw=1.6, solid_capstyle="round"); ax.plot(h["d"], yi, "o", ms=5, color=col, mec="white", mew=0.7, zorder=3)
    ax.text(0.012, yi, f"{h['better']} / {h['worse']}", va="center", fontsize=6.8, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
for t, r in zip(ax.get_yticklabels(), rows):
    if r[2]: t.set_color(BLUE); t.set_fontweight("bold")
ax.axvline(0, color=INK, lw=0.6); ax.set_xlim(-0.38, 0.075); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.text(0.012, len(rows) - 0.62, "better/worse", fontsize=8, fontweight="bold", color=INK)
ax.set_xlabel("Paired difference in false-positive tile\nfraction, IHC 0 slides (with − without)")
ax.set_title("IHC 0 slides (n = 138)", loc="left"); panel(ax, "a", x=-0.62)
# (b) positive fraction by IHC score
ax = fig.add_subplot(gs[0, 1]); r = S["025_frozen"]; xs = np.arange(4)
for k, col, off, lab in [(B, GREY, 0.09, "Without negative constraint"), (A, BLUE, -0.09, "SANC-MIL (with constraint)")]:
    v = r["posfrac"][k]; mu = [v[g][0] for g in range(4)]; lo = [v[g][1] for g in range(4)]; hi = [v[g][2] for g in range(4)]
    ax.bar(xs + off, mu, width=0.18, color=col, label=lab, zorder=2)
    ax.errorbar(xs + off, mu, yerr=[np.subtract(mu, lo), np.subtract(hi, mu)], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(["0\n(n=138)", "1+\n(n=102)", "2+\n(n=75)", "3+\n(n=103)"]); ax.set_xlabel("ADEL IHC score")
ax.set_ylabel("Mean fraction of tiles called positive"); ax.set_ylim(0, 1.12); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=6.6, ncol=1, handlelength=1.2); ax.set_title("Tile positivity, primary setting", loc="left"); panel(ax, "b", x=-0.2)
# (c) AUROC vs scale
ax = fig.add_subplot(gs[1, 0]); xs = np.array([0.22, 0.25, 0.28])
for k, col, lab, mk in [("ABMIL", AQUA, "Attention MIL", "s"), ("Stain-intensity rule", ORANGE, "Stain-intensity rule", "D"),
                        (B, GREY, "Without negative constraint", "o"), ("Membrane + cell features", "#7a5cc4", "Membrane + cell features", "^"), (A, BLUE, "SANC-MIL, status logit", "o")]:
    v = [S[f"{m}_frozen"]["auc"][k] for m, _ in SC]; mu = [a[0] for a in v]
    ax.fill_between(xs, [a[1] for a in v], [a[2] for a in v], color=col, alpha=0.12, lw=0)
    ax.plot(xs, mu, "-", color=col, lw=1.4, marker=mk, ms=4, label=lab, mec="white", mew=0.6)
EX = json.load(open(_os.path.join(ROOT, "results", "derived", "exp_auc.json")))
v = [EX[f"{m}_frozen"]["SANC-MIL"] for m, _ in SC]
ax.fill_between(xs, [a[1] for a in v], [a[2] for a in v], color=BLUE, alpha=0.12, lw=0)
ax.plot(xs, [a[0] for a in v], "--", color=BLUE, lw=1.4, marker="o", ms=4, mfc="white", mec=BLUE, mew=1.0, label="SANC-MIL, ordinal head (post hoc)")
ax.set_xticks(xs); ax.set_xticklabels(["0.22", "0.25\n(primary)", "0.28"]); ax.set_xlabel("Assumed native pixel size of ADEL scans (µm/px)")
ax.set_ylabel("Status AUROC (IHC 0/1+ vs 3+)"); ax.set_ylim(0.75, 1.0); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, columnspacing=1.0, handlelength=1.8)
ax.set_title("Status AUROC vs assumed pixel size", loc="left"); panel(ax, "c", x=-0.2)
# (d) strong membranous area by score
ax = fig.add_subplot(gs[1, 1]); st = S["025_frozen"]["strong"]
data = [np.asarray(st[g]) * 100 for g in range(4)]
bp = ax.boxplot(data, positions=range(4), widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.1),
                boxprops=dict(facecolor="#dbe8f8", edgecolor=BLUE, lw=0.8), whiskerprops=dict(color=BLUE, lw=0.8), capprops=dict(color=BLUE, lw=0.8))
for g in range(4):
    jx = g + rng.uniform(-0.17, 0.17, len(data[g])); ax.scatter(jx, data[g], s=3, color=BLUE, alpha=0.35, lw=0, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(["0", "1+", "2+", "3+"]); ax.set_xlabel("ADEL IHC score")
ax.set_ylabel("Strong membranous DAB area (% of image area)"); ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(-0.02, 40)
ax.set_yticks([0, 0.1, 1, 10]); ax.set_yticklabels(["0", "0.1", "1", "10"])
rs = [S[f"{m}_frozen"]["rho_strong"] for m, _ in SC]
ax.text(0.02, 0.97, f"Spearman ρ = {rs[1]:.2f}\n(0.22: {rs[0]:.2f}; 0.28: {rs[2]:.2f})", transform=ax.transAxes, va="top", fontsize=6.8, color=INK)
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title("Membrane student, no retraining", loc="left"); panel(ax, "d", x=-0.27)
save(fig, "Fig10_external_ADEL")
print("ok")
