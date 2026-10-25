# ===================== CELL E1. Extended external analysis (no re-run of models; takes about 1–2 min) =====================
# Reads the per-slide files written by cells C and D and adds the analyses that the paper reports beyond the pre-specified ones:
#   (1) status AUROC of BOTH outputs of each MIL model: the additive status logit and the ordinal head's expected IHC score,
#       with stratified bootstrap 95% CIs, and paired bootstrap tests (Holm-corrected within each family);
#   (2) subgroup AUROCs (IHC 0 vs 3+, IHC 1+ vs 3+);
#   (3) localisation on IHC 0 (H1), on IHC 1+ (secondary), and positive coverage in IHC 3+;
#   (4) every analysis with and without the 48 slides re-processed by cell B2.
# Output: tables/external_adel_extended.csv (long format) + external_adel_extended_summary.txt + a zip to send back.
# Works after a kernel restart: it needs only the CSV files on disk.
import os, json, zipfile, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score

OUT = globals().get("OUT", Path("/kaggle/working/her2_wsl_v2"))
TAB = OUT / "tables"
DST = OUT / "ext_adel_regions"
NAMES = {"adel_mpp022": 0.22, "adel_mpp025": 0.25, "adel_mpp028": 0.28}
A, B = "SANC-MIL", "SANC without negative constraint"
N_BOOT, SEED = 2000, 2024

rec_path = DST / "recovered_slides.csv"
RECOVERED = set(pd.read_csv(rec_path).slide) if rec_path.exists() else set()
print(f"re-processed slides listed: {len(RECOVERED)}")

def _cmp(y, s):                       # pairwise comparison matrix: AUROC of any bootstrap sample = weighted mean of it (fast, exact)
    s = np.asarray(s, float); P, N = s[y == 1], s[y == 0]
    return (P[:, None] > N[None, :]) + 0.5 * (P[:, None] == N[None, :])

def _counts(r, n, size):
    return np.stack([np.bincount(r.integers(0, n, n), minlength=n) for _ in range(size)]).astype(float)

def strat_boot(y, s, n=N_BOOT, seed=SEED):
    r = np.random.default_rng(seed); C = _cmp(y, s); cp, cn = _counts(r, C.shape[0], n), _counts(r, C.shape[1], n)
    b = np.einsum("bi,ij,bj->b", cp, C, cn) / (C.shape[0] * C.shape[1])
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))

def paired_auc(y, sa, sb, n=N_BOOT, seed=SEED + 1):
    r = np.random.default_rng(seed); Ca, Cb = _cmp(y, sa), _cmp(y, sb)
    cp, cn = _counts(r, Ca.shape[0], n), _counts(r, Ca.shape[1], n); k = Ca.shape[0] * Ca.shape[1]
    d = (np.einsum("bi,ij,bj->b", cp, Ca, cn) - np.einsum("bi,ij,bj->b", cp, Cb, cn)) / k
    return float(Ca.mean() - Cb.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), \
           float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r_, i in enumerate(o): run = max(run, min(1.0, (len(p) - r_) * p[i])); adj[i] = run
    return adj

def paired_fp(z):
    a, b = z[f"tileFP::{A}"].values, z[f"tileFP::{B}"].values; d = a - b
    r = np.random.default_rng(SEED); bs = [d[r.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(z), with_=a.mean(), without=b.mean(), diff=d.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                p=p, better=int((d < 0).sum()), worse=int((d > 0).sum()),
                gt09_with=float((z[f"tileMax::{A}"] > 0.9).mean()), gt09_without=float((z[f"tileMax::{B}"] > 0.9).mean()))

rows = []
def add(**k): rows.append(k)
for name, mpp in NAMES.items():
    f = TAB / f"external_{name}_per_item.csv"
    if not f.exists(): print("missing:", f); continue
    D = pd.read_csv(f); D["slide"] = D.item_id.str.extract(r"^(HER2_\d+)")[0]; D["reprocessed"] = D.slide.isin(RECOVERED)
    for mode, Dm in D.groupby("mode"):
        for subset, d in [("all", Dm), ("without_reprocessed", Dm[~Dm.reprocessed])]:
            base = dict(dataset=name, assumed_mpp=mpp, stain=mode, subset=subset)
            # ---- status: 0/1+ vs 3+ (2+ has no ISH) ----
            st = d[d.ihc_score != 2]; y = (st.ihc_score == 3).astype(int).values
            outputs = {f"{m} | status logit": st[f"logit::{m}"].values for m in [A, B, "ABMIL", "Stain-intensity rule", "Membrane + cell features"] if f"logit::{m}" in st}
            outputs.update({f"{m} | ordinal head (expected score)": st[f"exp_score::{m}"].values for m in [A, B] if f"exp_score::{m}" in st})
            outputs["Membrane student | strong membranous DAB area"] = st["stu_mem_strong_frac"].values
            for k, s in outputs.items():
                ok = np.isfinite(s)
                lo, hi = strat_boot(y[ok], s[ok])
                add(**base, family="status_AUROC", measure=k, value=roc_auc_score(y[ok], s[ok]), lo=lo, hi=hi, n_pos=int(y[ok].sum()), n_neg=int((1 - y[ok]).sum()))
            for g in (0, 1):
                sub = d[d.ihc_score.isin([g, 3])]; yy = (sub.ihc_score == 3).astype(int).values
                for m in (A, B):
                    for out, col in (("status logit", f"logit::{m}"), ("ordinal head", f"exp_score::{m}")):
                        add(**base, family=f"subgroup_AUROC_{g}v3", measure=f"{m} | {out}", value=roc_auc_score(yy, sub[col]))
            # ---- paired AUROC tests (one Holm family per dataset/stain/subset) ----
            tests = [(f"{A}: ordinal head − status logit", st[f"exp_score::{A}"].values, st[f"logit::{A}"].values),
                     (f"{A} − {B}: status logit", st[f"logit::{A}"].values, st[f"logit::{B}"].values),
                     (f"{A} − {B}: ordinal head", st[f"exp_score::{A}"].values, st[f"exp_score::{B}"].values)]
            res = [paired_auc(y, a, b) for _, a, b in tests]; ph = holm([r_[3] for r_ in res])
            for (lab, _, _), r_, h in zip(tests, res, ph):
                add(**base, family="paired_dAUROC", measure=lab, value=r_[0], lo=r_[1], hi=r_[2], p=r_[3], p_holm=h)
            # ---- localisation ----
            for g, fam in ((0, "H1_IHC0"), (1, "FP_IHC1plus")):
                z = d[d.ihc_score == g]
                if len(z) >= 5:
                    r_ = paired_fp(z)
                    add(**base, family=fam, measure="false-positive tile fraction, with − without", value=r_["diff"], lo=r_["lo"], hi=r_["hi"], p=r_["p"],
                        n=r_["n"], mean_with=r_["with_"], mean_without=r_["without"], better=r_["better"], worse=r_["worse"],
                        any_tile_gt09_with=r_["gt09_with"], any_tile_gt09_without=r_["gt09_without"])
            t3 = d[d.ihc_score == 3]
            add(**base, family="coverage_IHC3", measure="mean fraction of tiles positive, with / without", value=t3[f"tileFP::{A}"].mean(), value_B=t3[f"tileFP::{B}"].mean())
            # ---- ordinal agreement ----
            for m in (A, B):
                e = d[f"exp_score::{m}"]
                add(**base, family="ordinal_agreement", measure=m, value=stats.spearmanr(d.ihc_score, e).correlation,
                    qwk=cohen_kappa_score(d.ihc_score, np.clip(np.rint(e), 0, 3).astype(int), weights="quadratic"))

R = pd.DataFrame(rows)
R.to_csv(TAB / "external_adel_extended.csv", index=False)

# ---- readable summary (primary setting first) ----
lines = []
for (ds, stn, sub), g in R.groupby(["dataset", "stain", "subset"], sort=False):
    lines.append(f"\n===== {ds} | stain {stn} | {sub} =====")
    for _, r in g[g.family == "status_AUROC"].iterrows(): lines.append(f"  AUROC {r.measure:<55s} {r.value:.3f} ({r.lo:.3f}–{r.hi:.3f})")
    for _, r in g[g.family == "paired_dAUROC"].iterrows(): lines.append(f"  ΔAUROC {r.measure:<54s} {r.value:+.3f} ({r.lo:+.3f} to {r.hi:+.3f}) Holm p={r.p_holm:.4f}")
    for _, r in g[g.family.isin(["H1_IHC0", "FP_IHC1plus"])].iterrows():
        lines.append(f"  {r.family:<12s} with {r.mean_with:.3f} vs without {r.mean_without:.3f}; diff {r.value:+.3f} ({r.lo:+.3f} to {r.hi:+.3f}); p={r.p:.1e}; better/worse {int(r.better)}/{int(r.worse)}")
txt = "\n".join(lines); print(txt)
(TAB / "external_adel_extended_summary.txt").write_text(txt)
zp = OUT.parent / "external_adel_extended.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in (TAB / "external_adel_extended.csv", TAB / "external_adel_extended_summary.txt"): zf.write(p, p.name)
print(f"\nSEND THIS FILE: {zp}")
