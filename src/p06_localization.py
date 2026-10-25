# %% [markdown]
# ## 11. Localisation without reference masks: one grounded measure and several proxies
# * **Grounded:** the instance false-positive rate in **test IHC-0 cores**. A pathologist found no 3+-type staining there, so any tile with high
#   positivity is an error. Attention-only models (ABMIL, TransMIL) give no positivity score, so this does not apply to them. That gap is itself a
#   finding.
# * **Proxy: stability.** The mean pairwise Spearman correlation of tile scores across the 5 fold models, and the Jaccard overlap of the top-10% tiles.
# * **Proxy: faithfulness to the model.** Remove the top-scored tiles and measure the drop in P(HER2+), compared with removing random tiles.
#   This shows the map reflects the model's reasoning. It does **not** show the map is biologically correct.
# * **Proxy: stain concordance.** The Spearman correlation between tile positivity and membranous DAB at moderate or higher, within positive cores.
#   It is shown for SANC *without* the stain features, to avoid circularity.
#
# None of these establishes spatial accuracy.

# %%
def per_fold_tile_scores(res, kind="inst"):
    """res: a CV result dict (or a variant name in MIL_RES). Returns list over test bags of array (folds, n_tiles)."""
    if isinstance(res, str): res = MIL_RES[res]
    runs = res["test"]
    if runs[0][0][kind] is None: return None
    return [np.stack([runs[k][b][kind] for k in range(len(runs))]) for b in range(len(TE))]

neg0 = np.where(TE.ihc_score.values == 0)[0]; posb = np.where(TE.her2_status.values == 1)[0]
ihc3 = np.where(TE.ihc_score.values == 3)[0]
rng = np.random.default_rng(CFG.SEED)

def fold_stability(s_, min_tiles=5):
    """Mean pairwise Spearman rho and top-10% Jaccard between the fold models' tile scores of one bag.
    Written without scipy.stats.spearmanr on a matrix: some SciPy versions return a single NaN for the whole matrix when any
    input is constant or non-finite, which crashed the v2.1 run on Kaggle. Non-finite tiles are dropped; constant folds -> NaN."""
    s_ = np.asarray(s_, np.float64)
    if s_.ndim != 2 or s_.shape[0] < 2: return np.nan, np.nan
    s_ = s_[:, np.isfinite(s_).all(0)]
    if s_.shape[1] < min_tiles: return np.nan, np.nan
    R = stats.rankdata(s_, axis=1); R = R - R.mean(1, keepdims=True); nr = np.sqrt((R ** 2).sum(1))
    iu = np.triu_indices(len(R), 1); num = (R @ R.T)[iu]; den = (nr[:, None] * nr[None, :])[iu]
    rho = np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)) if (den > 0).any() else np.nan
    k = max(1, int(0.1 * s_.shape[1])); tops = [set(np.argsort(-x, kind="stable")[:k]) for x in s_]
    jac = np.mean([len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]])
    return float(rho), float(jac)

def _fin(x): x = np.asarray(x, np.float64); return x[np.isfinite(x)]

def loc_metrics(res, name):
    """Grounded check (IHC-0 tile false positives), grounded-direction check (IHC 3+ detection), and proxies."""
    inst = per_fold_tile_scores(res, "inst"); attn = per_fold_tile_scores(res, "attn")
    sc = inst if inst is not None else attn
    row = {"method": name, "score_type": "positivity" if inst is not None else ("attention" if attn is not None else "none")}
    if sc is None: return row
    row["nonfinite_tile_scores"] = int(sum((~np.isfinite(s_)).sum() for s_ in sc))   # should be 0; reported, never hidden
    rhos, jac = [], []
    for s_ in sc:
        r_, j_ = fold_stability(s_)
        rhos.append(r_); jac.append(j_)
    row["fold_stability_spearman"] = float(np.nanmean(rhos)) if np.isfinite(rhos).any() else np.nan
    row["top10pct_jaccard"] = float(np.nanmean(jac)) if np.isfinite(jac).any() else np.nan
    if inst is not None:
        P = [sig(x.mean(0)) for x in inst]
        row["IHC0_tile_FP_rate@0.5"] = float(np.mean(np.concatenate([P[b] > 0.5 for b in neg0]))) if len(neg0) else np.nan
        if len(neg0):   # core-level bootstrap CI of the pooled tile FP rate (cores, not tiles, are the independent units)
            fp_n = np.array([np.sum(P[b] > 0.5) for b in neg0]); tl_n = np.array([len(P[b]) for b in neg0]); rb = np.random.default_rng(0)
            bs = [fp_n[i].sum() / max(tl_n[i].sum(), 1) for i in (rb.integers(0, len(neg0), len(neg0)) for _ in range(1000))]
            row["IHC0_tile_FP_rate@0.5_lo"], row["IHC0_tile_FP_rate@0.5_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        oof_neg = [j for j in np.where((TR.ihc_score.values == 0))[0] if res["oof"][j] is not None and res["oof"][j].get("inst") is not None]
        if oof_neg:   # the same grounded check on ~1,300 out-of-fold IHC-0 TRAINING cores (scored by the fold model that never saw them)
            row["OOF_IHC0_tile_FP_rate@0.5"] = float(np.mean(np.concatenate([sig(res["oof"][j]["inst"]) > 0.5 for j in oof_neg])))
            row["OOF_IHC0_cores_with_any_tile>0.9"] = float(np.mean([np.any(sig(res["oof"][j]["inst"]) > 0.9) for j in oof_neg]))
        row["IHC0_tile_FP_rate@0.9"] = float(np.mean(np.concatenate([P[b] > 0.9 for b in neg0]))) if len(neg0) else np.nan
        row["IHC0_cores_with_any_tile>0.9"] = float(np.mean([np.any(P[b] > 0.9) for b in neg0])) if len(neg0) else np.nan
        row["IHC3_cores_detected(any tile>0.5)"] = float(np.mean([np.any(P[b] > 0.5) for b in ihc3])) if len(ihc3) else np.nan
        row["IHC3_median_pos_tile_frac"] = float(np.median([np.mean(P[b] > 0.5) for b in ihc3])) if len(ihc3) else np.nan
        if len(ihc3) and len(neg0):
            t3 = _fin(np.concatenate([P[b] for b in ihc3])); t0 = _fin(np.concatenate([P[b] for b in neg0]))
            row["tile_AUROC_IHC3_vs_IHC0"] = float(roc_auc_score(np.r_[np.ones(len(t3)), np.zeros(len(t0))], np.r_[t3, t0])) if len(t3) and len(t0) else np.nan
        conc = []
        for b in posb:
            mm = BAGS[TE.image_id[b]]["hand_raw"][:, HAND_NAMES.index("mem_mod")]
            v = inst[b].mean(0); ok = np.isfinite(v) & np.isfinite(mm)
            if ok.sum() > 5 and mm[ok].std() > 0 and v[ok].std() > 0: conc.append(stats.spearmanr(v[ok], mm[ok]).correlation)
        row["stain_concordance_rho(pos cores)"] = float(np.nanmean(conc)) if conc else np.nan
    return row

LOC_ROWS = []
for n, r in {**MIL_RES, **CAM_RES, **ENC_RES}.items():
    try: LOC_ROWS.append(loc_metrics(r, n))
    except Exception as e:      # one failing method must not stop the run; the failure is recorded in the table
        print(f"!! localisation metrics failed for {n}: {type(e).__name__}: {e}"); LOC_ROWS.append({"method": n, "score_type": f"ERROR: {type(e).__name__}"})
LOC = pd.DataFrame(LOC_ROWS)
_bad = LOC[LOC.get("nonfinite_tile_scores", pd.Series(0, index=LOC.index)).fillna(0) > 0]
if len(_bad): print("!! non-finite tile scores (excluded from rank metrics):\n", _bad[["method", "nonfinite_tile_scores"]].to_string(index=False))

# deletion faithfulness (model-centric) for the proposed model and ABMIL
@torch.no_grad()
def deletion_curve(name, frac_list=(0.0, 0.05, 0.1, 0.2, 0.3)):
    T = CLS[CLS.method == name].temp.iloc[0]
    models = fold_models(name)
    sc = per_fold_tile_scores(MIL_RES[name], "inst")
    sc = sc if sc is not None else per_fold_tile_scores(MIL_RES[name], "attn")
    top_drop, rnd_drop = [], []
    for b in posb:
        s = sc[b].mean(0); n = len(s); bp = TEP[b]
        e_full, h_full, _, _ = STORE.batch([bp])
        def prob(keep_idx):
            e, h = e_full[:, keep_idx], h_full[:, keep_idx]; mk = torch.ones(1, len(keep_idx), dtype=torch.bool, device=DEVICE)
            return float(np.mean([sig(m(e, h, mk)["logit"].item() / T) for m in models]))
        base = prob(np.arange(n)); t_row, r_row = [], []
        for f in frac_list[1:]:
            k = max(1, int(f * n)); order = np.argsort(-s)
            t_row.append(base - prob(np.sort(order[k:]))) if n - k > 0 else t_row.append(np.nan)
            r_row.append(np.mean([base - prob(np.sort(rng.choice(n, n - k, replace=False))) for _ in range(5)]) if n - k > 0 else np.nan)
        top_drop.append(t_row); rnd_drop.append(r_row)
    for m in models: del m
    return np.array(frac_list[1:]), np.nanmean(top_drop, 0), np.nanmean(rnd_drop, 0)
DEL_PATH = CACHE / "deletion_curves.pkl"
DEL = load_pickle(DEL_PATH) if DEL_PATH.exists() else None
if DEL is None:
    DEL = {name: deletion_curve(name) for name in [PROPOSED, MAIN_BASELINE] if len(posb)}
    atomic_pickle(DEL, DEL_PATH)
for name, (fr, td, rd) in DEL.items():
    LOC.loc[LOC.method == name, "deletion_drop@10%_top_minus_random"] = float(np.interp(0.1, fr, td) - np.interp(0.1, fr, rd))
LOC.to_csv(TAB / "localization_proxies.csv", index=False); print(LOC.round(3).to_string(index=False))

# ---------------- variability over repeated CV seeds (the claims must survive re-training) ----------------
SEED_ROWS = []
for n, runs in SEED_RES.items():
    for sd, res in enumerate(runs):
        te_logit = np.mean([[d["logit"] for d in run] for run in res["test"]], 0)
        r = loc_metrics(res, n); r["seed"] = sd; r["test_AUROC"] = roc_auc_score(ystat_te, te_logit)
        r["OOF_AUROC"] = roc_auc_score(ystat_tr, [d["logit"] for d in res["oof"]]); SEED_ROWS.append(r)
SEEDS = pd.DataFrame(SEED_ROWS)
_keys = ["test_AUROC", "OOF_AUROC", "IHC0_tile_FP_rate@0.5", "OOF_IHC0_tile_FP_rate@0.5", "IHC0_cores_with_any_tile>0.9", "IHC3_cores_detected(any tile>0.5)",
         "tile_AUROC_IHC3_vs_IHC0", "fold_stability_spearman"]
_keys = [k for k in _keys if k in SEEDS]
SEED_SUMMARY = SEEDS.groupby("method")[_keys].agg(["mean", "std"]).round(4)
SEEDS.to_csv(TAB / "seed_runs.csv", index=False); SEED_SUMMARY.to_csv(TAB / "seed_variability.csv")
print(f"Mean ± SD over {CFG.N_SEEDS} CV seeds:\n", SEED_SUMMARY.to_string())

# ---------------- H1: does the negative-bag constraint reduce false-positive evidence in IHC-0 cores? ----------------
# Paired per-core comparison (same cores, same folds): Wilcoxon signed-rank test + bootstrap 95% CI of the mean difference,
# on the independent test set AND on out-of-fold IHC-0 training cores, repeated for every CV seed.
def core_fp(res, split, thr=0.5):
    if split == "test":
        sc = per_fold_tile_scores(res, "inst")
        return None if sc is None else np.array([np.mean(sig(sc[b].mean(0)) > thr) for b in neg0])
    idx = np.where(TR.ihc_score.values == 0)[0]
    if res["oof"][idx[0]].get("inst") is None: return None
    return np.array([np.mean(sig(res["oof"][j]["inst"]) > thr) for j in idx])
def paired_test(a, b, n_boot=2000):
    d = a - b; rb = np.random.default_rng(1)
    bs = [d[rb.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    try: p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    except ValueError: p = 1.0
    return dict(n_cores=len(d), mean_A=a.mean(), mean_B=b.mean(), diff=d.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                wilcoxon_p=p, cores_A_better=int((d < 0).sum()), cores_B_better=int((d > 0).sum()))
H1_PAIRS = [(PROPOSED, "SANC − negative constraint"), ("SANC − stain features", "SANC − negative constraint"), (PROPOSED, "MaxPool")]
H1_ROWS = []
for A, B in H1_PAIRS:
    for sd in range(min(len(SEED_RES.get(A, [])), len(SEED_RES.get(B, [])))):
        for split in ("test", "oof"):
            a, b = core_fp(SEED_RES[A][sd], split), core_fp(SEED_RES[B][sd], split)
            if a is not None and b is not None: H1_ROWS.append(dict(A=A, B=B, seed=sd, split=split, **paired_test(a, b)))
if CAM_RES:
    a, b = core_fp(MIL_RES[PROPOSED], "test"), core_fp(CAM_RES["Grad-CAM CNN (per-image normalised CAM)"], "test")
    H1_ROWS.append(dict(A=PROPOSED, B="Grad-CAM CNN (per-image normalised CAM)", seed=0, split="test", **paired_test(a, b)))
H1 = pd.DataFrame(H1_ROWS); atomic_csv(H1, TAB / "h1_negative_constraint_tests.csv", index=False)
print("H1 paired per-core tests (FP fraction of tiles per IHC-0 core; A − B < 0 favours A):\n", H1.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
_show = [m for m in [MAIN_BASELINE, "MaxPool", "SANC − negative constraint", "SANC − stain features", PROPOSED] if m in SEEDS.method.unique()]
for j, (col, ttl) in enumerate([("test_AUROC", "Test AUROC (status)"), ("IHC0_tile_FP_rate@0.5", "IHC-0 tile FP rate, test"),
                                ("OOF_IHC0_tile_FP_rate@0.5", "IHC-0 tile FP rate, OOF train")]):
    if col not in SEEDS: ax[j].axis("off"); continue
    for i, m in enumerate(_show):
        v = SEEDS[SEEDS.method == m][col].dropna().values
        if len(v): ax[j].errorbar(v.mean(), i, xerr=v.std() if len(v) > 1 else 0, fmt="o", color=PAL[i], ms=6, capsize=3)
        ax[j].scatter(v, np.full(len(v), i), s=10, color=PAL[i], alpha=0.5)
    ax[j].set_yticks(range(len(_show))); ax[j].set_yticklabels(_show if j == 0 else [], fontsize=8); ax[j].set_title(ttl)
fig.suptitle(f"Effect of the negative-bag constraint (mean ± SD over {CFG.N_SEEDS} CV seeds; dots = individual seeds)", fontsize=10)
savefig(fig, "fig11_negative_constraint_seeds")
section_time("localisation analysis")

# %%
# ---------------- heat-map figure: attention ≠ positivity; uncertainty shown explicitly ----------------
def tile_grid_map(iid, values, shape_lr):
    T, R = CFG.TILE, CFG.THUMB_REL; m = np.full(shape_lr, np.nan, np.float32)
    for (y, x), v in zip(BAGS[iid]["coords"], values):
        m[y // R:(y + T) // R, x // R:(x + T) // R] = v
    return m
def heatmap_figure(n_per=1):
    picks = []
    for g in ["0", "2-", "2+", "3"]:
        s = TE[TE.group == g]
        if len(s): picks += [s.index[i] for i in rng.choice(len(s), min(n_per, len(s)), replace=False)]
    inst_s = per_fold_tile_scores(PROPOSED, "inst"); att_a = per_fold_tile_scores(MAIN_BASELINE, "attn")
    cam_n = per_fold_tile_scores(CAM_RES["Grad-CAM CNN (per-image normalised CAM)"], "inst") if CAM_RES else None
    cols = ["core", "DAB OD", "membranous DAB ≥ moderate", "Grad-CAM (per-image norm.)", "ABMIL attention", "SANC P(positive tile)", "SANC uncertainty (fold SD)"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.3 * len(cols), 2.35 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, b in enumerate(picks):
        iid = TE.image_id[b]; mp = np.load(MAP_DIR / f"{iid}.npz"); th = cv2.cvtColor(cv2.imread(str(THUMB_DIR / f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
        shp = mp["dab"].shape
        tmask = cv2.resize(mp["tissue"].astype(np.uint8), (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST) > 0
        nan_out = lambda a: np.where(tmask, a.astype(np.float32), np.nan)
        pr = sig(inst_s[b]); att = att_a[b].mean(0); att = att / (att.max() + 1e-9)
        camv = tile_grid_map(iid, sig(cam_n[b]).mean(0), shp) if cam_n is not None else np.full(shp, np.nan, np.float32)
        panels = [(th, None), (nan_out(mp["dab"]), "magma"), (nan_out(mp["mem_mod"]), "magma"), (camv, "viridis"),
                  (tile_grid_map(iid, att, shp), "viridis"), (tile_grid_map(iid, pr.mean(0), shp), "RdBu_r"), (tile_grid_map(iid, pr.std(0), shp), "Greys")]
        for c, (img, cm) in enumerate(panels):
            ax = axes[r, c]; ax.axis("off")
            if cm is None: ax.imshow(img)
            else:
                ax.imshow(th, alpha=0.35)
                kw = dict(vmin=0, vmax=1) if c in (3, 4, 5) else (dict(vmin=0, vmax=0.35) if c == 6 else {})
                if c == 1: kw = dict(vmin=0, vmax=CALIB["t_strong"] * 1.5)
                if c == 2: kw = dict(vmin=0, vmax=max(0.05, float(np.nanpercentile(img, 99.5))))
                ax.imshow(cv2.resize(img, (th.shape[1], th.shape[0]), interpolation=cv2.INTER_NEAREST), cmap=cm, alpha=0.75, **kw)
            if r == 0: ax.set_title(cols[c], fontsize=8)
        axes[r, 0].text(5, 25, f"IHC {TE.group[b]}  P={pu['pm'][b]:.2f}", color="k", fontsize=7, bbox=dict(fc="w", ec="none", alpha=0.8))
    savefig(fig, "fig08_localization_maps")
heatmap_figure()
