# %% [markdown]
# ## 5. Quality control: flag rather than drop, and test whether QC is itself confounded by class
# The earlier pipeline dropped 49% of the training cores with a global Laplacian-variance threshold. That statistic mostly measures the tissue
# amount and the DAB contrast, so the threshold dropped 52% of IHC-0 cores but only 21% of IHC-3 cores.
#
# This notebook handles QC differently:
# * Focus is measured **per tissue tile** and expressed as a robust z-score (median and MAD).
# * Flagged cores are **kept**. A sensitivity analysis later re-scores the test set without them.
# * The only hard exclusion is a core with fewer than `MIN_TILES_KEEP` tissue tiles.

# %%
QC = pd.DataFrame([json.loads(np.load(FEAT_DIR / f"{i}.npz")["qc"].item()) for i in ALL.image_id if (FEAT_DIR / f"{i}.npz").exists()])
ALLQ = ALL.merge(QC, on="image_id", how="left")
def robust_z(x, ref):
    med = np.nanmedian(ref); mad = 1.4826 * np.nanmedian(np.abs(ref - med)) + 1e-9
    return (x - med) / mad
tr_mask = ALLQ.split == "train"
lf = np.log(ALLQ.focus_med.clip(lower=1e-6))
ALLQ["focus_z"] = robust_z(lf, lf[tr_mask])
ALLQ["stain_z"] = robust_z(ALLQ.dab_hue_cos, ALLQ.dab_hue_cos[tr_mask])
# Train: cores with too little tissue are excluded from model fitting. Test: NO core is dropped (intention-to-diagnose);
# a test core with < MIN_TILES_KEEP tiles is kept, flagged low-tissue, and routed to "refer". Only empty cores (0 tiles) are dropped.
ALLQ["exclude"] = np.where(ALLQ.split == "train", ALLQ.n_tiles.fillna(0) < CFG.MIN_TILES_KEEP, ALLQ.n_tiles.fillna(0) < 1)
ALLQ["few_tiles"] = ALLQ.n_tiles.fillna(0) < CFG.MIN_TILES_KEEP
ALLQ["flag_blur"] = ALLQ.focus_z < -3.5
ALLQ["flag_dark"] = ALLQ.dark_frac > 0.15
ALLQ["flag_stain"] = ALLQ.stain_z < -4
ALLQ["flag_lowtissue"] = (ALLQ.tissue_mm2 < 0.10) | ALLQ.few_tiles
ALLQ["qc_flag"] = ALLQ[["flag_blur", "flag_dark", "flag_stain", "flag_lowtissue"]].any(axis=1)
qc_tab = ALLQ.groupby(["split", "group"])[["exclude", "flag_blur", "flag_dark", "flag_stain", "flag_lowtissue", "qc_flag"]].mean().round(3)
qc_tab.to_csv(TAB / "qc_flags_by_group.csv"); print(qc_tab)
trq = ALLQ[tr_mask & ~ALLQ.exclude]
ct = pd.crosstab(trq.group, trq.qc_flag)
chi_p = stats.chi2_contingency(ct)[1] if ct.shape == (ct.shape[0], 2) and ct.values.min() >= 0 and ct.shape[1] == 2 else np.nan
rho_old = stats.spearmanr(trq.ihc_score, trq.global_lapvar, nan_policy="omit")
rho_new = stats.spearmanr(trq.ihc_score, trq.focus_z, nan_policy="omit")
old_thr = np.nanmedian(trq.global_lapvar)
old_fail = trq.assign(f=trq.global_lapvar < old_thr).groupby("group").f.mean().reindex(GROUP_ORDER)
qc_summary = dict(chi2_p_new_flags_vs_group=chi_p, spearman_old_globalLapVar_vs_score=(rho_old.correlation, rho_old.pvalue),
                  spearman_new_tileFocus_vs_score=(rho_new.correlation, rho_new.pvalue),
                  old_style_fail_rate_by_group_at_median_threshold=old_fail.round(3).to_dict(),
                  n_excluded=int(ALLQ.exclude.sum()), n_flagged=int(ALLQ.qc_flag.sum()))
# edge artefact: membrane-shaped DAB density at the core rim vs interior (paired, per core)
ea = ALLQ[~ALLQ.exclude & ALLQ.mem_weak_rim.notna()]
w = stats.wilcoxon(ea.mem_weak_rim, ea.mem_weak_interior) if len(ea) > 10 else None
qc_summary["edge_artefact_rim_vs_interior_median"] = (float(ea.mem_weak_rim.median()), float(ea.mem_weak_interior.median()))
qc_summary["edge_artefact_wilcoxon_p"] = float(w.pvalue) if w is not None else np.nan
# temporal drift train -> test on IHC-0 cores (background staining & illumination)
drift = {}
for c in ["I0", "dab_hue_cos", "dab_mean_core", "focus_z"]:
    a = ALLQ[(ALLQ.split == "train") & (ALLQ.ihc_score == 0)][c].dropna(); b = ALLQ[(ALLQ.split == "test") & (ALLQ.ihc_score == 0)][c].dropna()
    if len(a) > 3 and len(b) > 3:
        drift[c] = dict(train_median=float(a.median()), test_median=float(b.median()), mannwhitney_p=float(stats.mannwhitneyu(a, b).pvalue))
qc_summary["train_test_drift_IHC0"] = drift
save_json(qc_summary, TAB / "qc_summary.json"); print(json.dumps(qc_summary, indent=1, default=str))

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, col, title in [(axes[0], "global_lapvar", "Old criterion: global Laplacian variance"),
                       (axes[1], "focus_z", "New: tile-level focus (robust z)")]:
    data = [trq[trq.group == g][col].dropna().values for g in GROUP_ORDER]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55); ax.set_xticks(range(1, len(GROUP_ORDER) + 1)); ax.set_xticklabels(GROUP_ORDER)
    for patch, g in zip(bp["boxes"], GROUP_ORDER): patch.set_facecolor(GROUP_COL[g]); patch.set_alpha(0.55)
    ax.set_title(title); ax.set_xlabel("IHC group (2-/2+ = FISH)")
axes[0].axhline(old_thr, color="#52514e", ls="--", lw=1)
axes[1].axhline(-3.5, color="#52514e", ls="--", lw=1)
axes[2].scatter(ea.mem_weak_interior, ea.mem_weak_rim, s=6, alpha=0.5, color=PAL[0])
mx = float(np.nanmax([ea.mem_weak_interior.max(), ea.mem_weak_rim.max(), 1e-3]))
axes[2].plot([0, mx], [0, mx], color="#52514e", lw=1, ls="--"); axes[2].set_xlabel("membranous DAB fraction: interior")
axes[2].set_ylabel("rim"); axes[2].set_title("Core-edge staining (edge artefact check)")
savefig(fig, "fig04_qc")

# %%
# ---------------- near-duplicate detection (thumbnail dHash) ----------------
hq = ALLQ[ALLQ.dhash.notna()].reset_index(drop=True)
bits = np.array([[int(c, 16) >> s & 1 for c in h for s in (3, 2, 1, 0)] for h in hq.dhash], np.float32)
ham = bits.sum(1)[:, None] + bits.sum(1)[None, :] - 2 * bits @ bits.T
iu = np.triu_indices(len(hq), 1)
close = [(hq.image_id[i], hq.image_id[j], int(ham[i, j])) for i, j in zip(*iu) if ham[i, j] <= 12]
cross = [c for c in close if c[0].split("__")[0] != c[1].split("__")[0]]
print(f"near-duplicate thumbnail pairs: {len(close)} (across train/test: {len(cross)})")
pd.DataFrame(close, columns=["a", "b", "hamming"]).to_csv(TAB / "near_duplicates.csv", index=False)

# %% [markdown]
# ## 6. Load the bags, normalise the features, and build proxy patient groups
# **Leakage.** Up to about 1,320 of the training cores come from 165 tumours, with 8 cores each, and there are no patient IDs. Random cross-validation
# therefore mixes cores of the same tumour across folds. As a mitigation, this section builds **proxy groups**: average-linkage clusters of
# core-level morphology and stain descriptors, sized to the expected number of patients. `StratifiedGroupKFold` then keeps each proxy group inside
# one fold.
#
# This is imperfect. Section 8 measures the effect by comparing grouped and random CV. The **official temporally independent test set is the
# primary evaluation.**

# %%
KEEP = ALLQ[~ALLQ.exclude & ALLQ.n_tiles.notna()].reset_index(drop=True)
TR = KEEP[KEEP.split == "train"].reset_index(drop=True)
TE = KEEP[KEEP.split == "test"].reset_index(drop=True)
print(f"usable cores: train {len(TR)} | test {len(TE)}  (excluded: {int(ALLQ.exclude.sum())})")

def transform_hand(h):
    h = h.copy().astype(np.float32)
    fi = HAND_NAMES.index("focus")
    h[:, fi] = np.log1p(np.clip(h[:, fi], 0, None))
    o = [i for i in range(h.shape[1]) if i != fi]
    h[:, o] = np.sqrt(np.clip(h[:, o], 0, None))
    return h

BAGS = {}
for iid in tqdm(KEEP.image_id, desc="load bags"):
    z = np.load(FEAT_DIR / f"{iid}.npz")
    BAGS[iid] = dict(emb=z["emb"], hand_raw=z["hand"], hand=transform_hand(z["hand"]), coords=z["coords"])
    if "emb2" in z.files: BAGS[iid]["emb2"] = z["emb2"]
HAS_EMB2 = all("emb2" in b for b in BAGS.values())
_ht = np.concatenate([BAGS[i]["hand"] for i in TR.image_id])
HAND_MU, HAND_SD = _ht.mean(0), _ht.std(0) + 1e-6
for b in BAGS.values():
    b["hand_z"] = ((b["hand"] - HAND_MU) / HAND_SD).astype(np.float32)
EMB_DIM = next(iter(BAGS.values()))["emb"].shape[1]

# ---------- proxy patient groups ----------
desc_e = np.stack([BAGS[i]["emb"].astype(np.float32).mean(0) for i in TR.image_id])
desc_e /= np.linalg.norm(desc_e, axis=1, keepdims=True) + 1e-9
desc_h = np.stack([BAGS[i]["hand_z"].mean(0) for i in TR.image_id])
desc_h = (desc_h - desc_h.mean(0)) / (desc_h.std(0) + 1e-6)
desc = np.concatenate([desc_e, 0.5 * desc_h / np.sqrt(desc_h.shape[1])], 1)
n_groups = int(np.clip(round(len(TR) * CFG.EXPECTED_PATIENTS / CFG.EXPECTED_CORES), 2, len(TR) - 1))
try:
    agg = AgglomerativeClustering(n_clusters=n_groups, metric="cosine", linkage="average")
except TypeError:
    agg = AgglomerativeClustering(n_clusters=n_groups, affinity="cosine", linkage="average")
TR["proxy_group"] = agg.fit_predict(desc)
sizes = TR.proxy_group.value_counts()
def within_agree(g, y):
    d = pd.DataFrame({"g": g, "y": y}); tot = 0; agree = 0
    for _, s in d.groupby("g"):
        n = len(s)
        if n > 1:
            vc = s.y.value_counts(); agree += (vc * (vc - 1)).sum(); tot += n * (n - 1)
    return agree / max(tot, 1)
obs = within_agree(TR.proxy_group, TR.ihc_score)
perm = np.mean([within_agree(np.random.default_rng(k).permutation(TR.proxy_group.values), TR.ihc_score) for k in range(50)])
group_info = dict(n_groups=n_groups, size_median=float(sizes.median()), size_max=int(sizes.max()),
                  singleton_frac=float((sizes == 1).mean()), within_group_score_agreement=obs, permutation_expectation=perm)
save_json(group_info, TAB / "proxy_groups.json"); print(group_info)

ylab = TR.group.map({g: i for i, g in enumerate(GROUP_ORDER)}).values
FOLDS_PATH = CACHE / "folds.csv"
_prev_folds = pd.read_csv(FOLDS_PATH) if FOLDS_PATH.exists() else None
if _prev_folds is not None and set(_prev_folds.image_id) == set(TR.image_id):
    # restart: reuse the exact saved split so every cached model/result stays consistent with it
    TR = TR.drop(columns=[c for c in ("proxy_group", "fold", "fold_random") if c in TR]).merge(_prev_folds, on="image_id", how="left")
    print("folds: loaded saved split", FOLDS_PATH)
else:
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    TR["fold"] = -1
    for k, (_, va) in enumerate(sgkf.split(TR, ylab, TR.proxy_group)):
        TR.loc[va, "fold"] = k
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    TR["fold_random"] = -1
    for k, (_, va) in enumerate(skf.split(TR, ylab)):
        TR.loc[va, "fold_random"] = k
    atomic_csv(TR[["image_id", "proxy_group", "fold", "fold_random"]], FOLDS_PATH, index=False)
print(pd.crosstab(TR.fold, TR.group).reindex(columns=GROUP_ORDER, fill_value=0))

# %% [markdown]
# ## 7. Unsupervised tissue-type clusters, and a 10-minute expert naming task
# HER2 labels contain no tumour/stroma information, so tumour localisation can't be *learned* from them. Instead, tile embeddings are clustered and
# a gallery is exported. A pathologist fills in `cluster_naming_template.csv` (tumour epithelium / stroma / necrosis / mucin / lymphoid / artefact)
# and points `CFG.CLUSTER_NAMING_CSV` to it. The tumour-candidate maps and the tumour-restricted cell features then switch on automatically.
# Until that happens, every result is computed on **all tissue** and labelled as such.

# %%
# Single-threaded BLAS/OpenMP here: after PyTorch DataLoader workers have run, scikit-learn's PCA/k-means can
# dead-lock on Kaggle when two OpenMP runtimes are loaded. Single-threaded, this whole step takes seconds.
from threadpoolctl import threadpool_limits
t0 = time.time(); rng = np.random.default_rng(CFG.SEED)
_ids = list(TR.image_id); _n = np.array([len(BAGS[i]["emb"]) for i in _ids]); _tot = int(_n.sum())
_pick = np.sort(rng.choice(_tot, min(30000, _tot), replace=False)); _off = np.concatenate([[0], np.cumsum(_n)])
sub = np.stack([BAGS[_ids[b]]["emb"][j - _off[b]] for j, b in zip(_pick, np.searchsorted(_off, _pick, side="right") - 1)]).astype(np.float32)
with threadpool_limits(limits=1):
    pca = PCA(n_components=min(50, sub.shape[1], len(sub) - 1), svd_solver="randomized", random_state=CFG.SEED).fit(sub)
    print(f"[tissue clusters] PCA on {len(sub)} tiles: {time.time() - t0:.0f}s", flush=True)
    km = MiniBatchKMeans(n_clusters=CFG.N_TISSUE_CLUSTERS, random_state=CFG.SEED, batch_size=4096, n_init=3).fit(pca.transform(sub))
    print(f"[tissue clusters] k-means: {time.time() - t0:.0f}s", flush=True)
    for iid, b in BAGS.items():
        b["tc"] = km.predict(pca.transform(b["emb"].astype(np.float32))) if len(b["emb"]) else np.zeros(0, int)
print(f"[tissue clusters] all {len(BAGS)} cores assigned: {time.time() - t0:.0f}s", flush=True)
comp = pd.DataFrame([np.bincount(BAGS[i]["tc"], minlength=CFG.N_TISSUE_CLUSTERS) / max(len(BAGS[i]["tc"]), 1) for i in TR.image_id])
comp["group"] = TR.group.values
comp_g = comp.groupby("group").mean().reindex(GROUP_ORDER)
mem_by_c = np.zeros(CFG.N_TISSUE_CLUSTERS); n_by_c = np.zeros(CFG.N_TISSUE_CLUSTERS)
for i in TR.image_id:
    b = BAGS[i]; np.add.at(mem_by_c, b["tc"], b["hand_raw"][:, HAND_NAMES.index("mem_weak")]); np.add.at(n_by_c, b["tc"], 1)
tmpl = pd.DataFrame({"cluster": range(CFG.N_TISSUE_CLUSTERS), "n_tiles_train": n_by_c.astype(int),
                     "mean_membranous_DAB_frac": (mem_by_c / np.maximum(n_by_c, 1)).round(4),
                     **{f"share_in_IHC{g}": comp_g.loc[g].values.round(3) if g in comp_g.index and comp_g.loc[g].notna().all() else np.nan for g in GROUP_ORDER},
                     "tissue_type": "", "notes": ""})
tmpl.to_csv(TAB / "cluster_naming_template.csv", index=False)

def gallery(n_per=8):
    picks = defaultdict(list)
    ids = list(TR.image_id); rng.shuffle(ids)
    for iid in ids:
        b = BAGS[iid]
        for c in range(CFG.N_TISSUE_CLUSTERS):
            if len(picks[c]) < n_per:
                w = np.where(b["tc"] == c)[0]
                if len(w): picks[c].append((iid, b["coords"][rng.choice(w)]))
        if all(len(picks[c]) >= n_per for c in range(CFG.N_TISSUE_CLUSTERS)): break
    need = defaultdict(list)
    for c, lst in picks.items():
        for j, (iid, yx) in enumerate(lst): need[iid].append((c, j, yx))
    T = CFG.TILE; canvas = np.full((CFG.N_TISSUE_CLUSTERS * T, n_per * T, 3), 255, np.uint8)
    path_of = dict(zip(KEEP.image_id, KEEP.path))
    for iid, lst in tqdm(need.items(), desc="cluster gallery"):
        work = read_rgb(path_of[iid], CFG.WORK_DS)
        for c, j, (y, x) in lst: canvas[c * T:(c + 1) * T, j * T:(j + 1) * T] = work[y:y + T, x:x + T]
    fig, ax = plt.subplots(figsize=(n_per * 1.1, CFG.N_TISSUE_CLUSTERS * 1.1))
    ax.imshow(canvas); ax.set_yticks([c * T + T / 2 for c in range(CFG.N_TISSUE_CLUSTERS)])
    ax.set_yticklabels([f"cluster {c}" for c in range(CFG.N_TISSUE_CLUSTERS)]); ax.set_xticks([]); ax.grid(False)
    ax.set_title("Tile clusters for expert naming (fill tables/cluster_naming_template.csv)")
    savefig(fig, "fig05_cluster_gallery")
if not (FIG / "fig05_cluster_gallery.png").exists(): gallery()

TUMOUR_CLUSTERS = None
if CFG.CLUSTER_NAMING_CSV and Path(CFG.CLUSTER_NAMING_CSV).exists():
    nm = pd.read_csv(CFG.CLUSTER_NAMING_CSV)
    TUMOUR_CLUSTERS = set(nm[nm.tissue_type.astype(str).str.lower().str.contains("tum")].cluster.astype(int))
    print("expert-named tumour clusters:", TUMOUR_CLUSTERS)
else:
    print("No cluster naming supplied: tumour localisation stays UNVERIFIED; features use all tissue.")
