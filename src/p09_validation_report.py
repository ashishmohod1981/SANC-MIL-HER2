# %% [markdown]
# ## 17. True segmentation accuracy, available only once expert masks exist
# Point `CFG.REF_MASK_DIR` to a folder of binary PNG masks, one per worklist ROI: `<roi_id>.png` for the primary annotator and
# `<roi_id>__<initials>.png` for a second annotator. Whole-core masks (`<image stem>.png`) are also accepted.
#
# **Why the metric set is more than Dice.** HER2 membranes are 2–4 px wide at 0.46 µm/px. On such thin structures, plain pixel Dice
# punishes a correct line drawn 1 px to the side as if it were wrong. Expert masks are drawn as lines, which makes this worse. So the
# main metrics are the ones used for thin, curvilinear structures:
# * **clDice** (centreline Dice): is each centreline covered by the other mask?
# * **Tolerance F1 at 1 µm**: a pixel counts as correct if it lies within 1 µm of the other mask.
# * **Boundary F1 at 2 µm** and **HD95** (µm).
#
# Plain Dice and IoU are still reported, for comparability with other papers.
#
# **ROIs with no membrane** (common in IHC 0) have no defined Dice. They are scored separately as the predicted membrane area, which is a
# false-positive measure, so they never inflate or deflate the Dice.
#
# **Statistics.** The core is the independent unit: metrics are averaged per core, then given 95% bootstrap CIs over cores. Only the primary
# annotator's masks go into the main estimate.
#
# **Inter-observer benchmark.** When a second annotator has scored a subset, the code compares model-vs-annotator agreement with
# annotator-vs-annotator agreement on the same ROIs, with a paired bootstrap. Reviewers judge the model against that ceiling.

# %%
from skimage.morphology import skeletonize
def boundary(m): return m ^ ndi.binary_erosion(m, iterations=1, border_value=0)
def seg_metrics(pred, ref, mpp=WORK_MPP, tol_um=1.0, btol_um=2.0, empty_frac=0.005):
    pred, ref = pred.astype(bool), ref.astype(bool)
    out = dict(ref_frac=float(ref.mean()), pred_frac=float(pred.mean()))
    if not ref.any():   # membrane-free ROI: Dice undefined -> report false-positive area only
        out.update(ref_empty=True, pred_empty=bool(pred.mean() < empty_frac))
        return out
    out["ref_empty"] = False
    tp = (pred & ref).sum(); fp = (pred & ~ref).sum(); fn = (~pred & ref).sum()
    out.update(Dice=2 * tp / max(2 * tp + fp + fn, 1), IoU=tp / max(tp + fp + fn, 1), Sensitivity=tp / max(tp + fn, 1), Precision=tp / max(tp + fp, 1))
    if not pred.any():
        out.update(clDice=0.0, TolF1=0.0, BoundaryF1=0.0, HD95_um=np.nan); return out
    d_ref = ndi.distance_transform_edt(~ref) * mpp; d_pred = ndi.distance_transform_edt(~pred) * mpp
    tp_p = (d_ref[pred] <= tol_um).mean(); tp_r = (d_pred[ref] <= tol_um).mean()            # tolerance precision / recall
    out["TolF1"] = 2 * tp_p * tp_r / max(tp_p + tp_r, 1e-9)
    sp, sr = skeletonize(pred), skeletonize(ref)                                              # clDice (Shit et al., CVPR 2021)
    t_prec = (sp & ref).sum() / max(sp.sum(), 1); t_sens = (sr & pred).sum() / max(sr.sum(), 1)
    out["clDice"] = 2 * t_prec * t_sens / max(t_prec + t_sens, 1e-9)
    bp, br = boundary(pred), boundary(ref)
    dp = ndi.distance_transform_edt(~br)[bp] * mpp; dr = ndi.distance_transform_edt(~bp)[br] * mpp
    out["HD95_um"] = float(max(np.percentile(dp, 95), np.percentile(dr, 95)))
    prc, rec = (dp <= btol_um).mean(), (dr <= btol_um).mean(); out["BoundaryF1"] = 2 * prc * rec / max(prc + rec, 1e-9)
    return out

def _read_mask(p, shape=None):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    m = m > 0
    if shape is not None and m.shape != shape: m = cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return m

def parse_ref_name(stem):
    """'<roi_id>' -> (roi_id, None); '<roi_id>__<initials>' -> (roi_id, initials). ROI ids themselves end in '__roi<k>'."""
    parts = stem.split("__")
    if len(parts) >= 2 and not re.fullmatch(r"roi\d+", parts[-1]): return "__".join(parts[:-1]), parts[-1]
    return stem, None

def pred_lookup(stem):
    f = OUT / "annotation_worklist" / "pred_masks" / f"{stem}.png"
    if f.exists(): return _read_mask(f)
    f = TEST_OUT / f"{stem}_membrane_grade.png"
    return _read_mask(f) if f.exists() else None

METRICS = ["Dice", "IoU", "Sensitivity", "Precision", "clDice", "TolF1", "BoundaryF1", "HD95_um"]
def core_boot(df, cols, n=None, seed=0):
    core = df.groupby("core")[cols].mean(); r0 = np.random.default_rng(seed); n = n or CFG.N_BOOT
    bs = pd.DataFrame([core.iloc[r0.integers(0, len(core), len(core))].mean() for _ in range(n)])
    return pd.DataFrame({"mean_over_cores": core.mean(), "lo": bs.quantile(0.025), "hi": bs.quantile(0.975), "n_cores": len(core)})

if CFG.REF_MASK_DIR and Path(CFG.REF_MASK_DIR).exists():
    refs = {}
    for p in sorted(Path(CFG.REF_MASK_DIR).glob("*.png")):
        unit, annot = parse_ref_name(p.stem); refs.setdefault(unit, {})[annot or "primary"] = p
    rows, pair_rows = [], []
    for unit, ann in refs.items():
        pred = pred_lookup(unit)
        if pred is None: print(f"  no prediction found for {unit}; skipped"); continue
        core = unit.split("__roi")[0]
        for who, p in ann.items():
            ref = _read_mask(p, pred.shape)
            if ref is not None: rows.append(dict(unit=unit, core=core, annotator=who, **seg_metrics(pred, ref)))
        names = sorted(ann)                                   # annotator-vs-annotator on the same ROI (inter-observer ceiling)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = _read_mask(ann[names[i]], pred.shape), _read_mask(ann[names[j]], pred.shape)
                if a is not None and b is not None:
                    pair_rows.append(dict(unit=unit, core=core, pair=f"{names[i]} vs {names[j]}", **seg_metrics(a, b)))
    REFM = pd.DataFrame(rows); atomic_csv(REFM, TAB / "reference_mask_metrics_per_roi.csv", index=False)
    prim = REFM[(REFM.annotator == "primary") & ~REFM.ref_empty.astype(bool)]
    neg = REFM[(REFM.annotator == "primary") & REFM.ref_empty.astype(bool)]
    print(f"reference masks: {REFM.unit.nunique()} ROIs, {REFM.core.nunique()} cores, annotators: {sorted(REFM.annotator.unique())}")
    if len(prim):
        SUMM = core_boot(prim, [c for c in METRICS if c in prim])
        atomic_csv(SUMM.reset_index().rename(columns={"index": "metric"}), TAB / "reference_mask_metrics_summary.csv", index=False)
        print("Model vs primary expert, membrane-present ROIs (mean over cores, 95% bootstrap CI over cores):"); print(SUMM.round(3))
        _wl = OUT / "annotation_worklist" / "worklist.csv"
        if _wl.exists():
            g = prim.merge(pd.read_csv(_wl)[["roi_id", "ihc_group"]], left_on="unit", right_on="roi_id", how="left")
            print("by IHC group (median clDice / TolF1):\n", g.groupby("ihc_group")[["clDice", "TolF1"]].median().round(3))
    if len(neg):
        NEGS = dict(n_rois=len(neg), n_cores=neg.core.nunique(), median_pred_membrane_frac=float(neg.pred_frac.median()),
                    rois_with_pred_below_0_5pct=float(neg.pred_empty.mean()))
        save_json(NEGS, TAB / "reference_mask_membrane_free_rois.json"); print("membrane-free ROIs (false-positive check):", NEGS)
    PAIRS = pd.DataFrame(pair_rows)
    if len(PAIRS):
        PAIRS = PAIRS[~PAIRS.ref_empty.astype(bool)]; atomic_csv(PAIRS, TAB / "inter_observer_per_roi.csv", index=False)
        sec = [a for a in REFM.annotator.unique() if a != "primary"]
        both = PAIRS[PAIRS.pair.str.contains("primary")].unit.unique()
        cmp_rows = []
        for m_ in ["clDice", "TolF1", "Dice"]:
            mp_ = REFM[(REFM.annotator == "primary") & REFM.unit.isin(both)].groupby("core")[m_].mean()
            ms_ = REFM[REFM.annotator.isin(sec) & REFM.unit.isin(both)].groupby("core")[m_].mean()
            io_ = PAIRS[PAIRS.unit.isin(both)].groupby("core")[m_].mean()
            model_ = pd.concat([mp_, ms_], axis=1).mean(axis=1).reindex(io_.index)
            d = (model_ - io_).dropna(); r0 = np.random.default_rng(1)
            bs = [d.iloc[r0.integers(0, len(d), len(d))].mean() for _ in range(CFG.N_BOOT)] if len(d) else [np.nan]
            cmp_rows.append(dict(metric=m_, n_cores=len(d), model_vs_experts=float(model_.mean()), expert_vs_expert=float(io_.mean()),
                                 diff=float(d.mean()) if len(d) else np.nan, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975))))
        IOBS = pd.DataFrame(cmp_rows); atomic_csv(IOBS, TAB / "model_vs_interobserver.csv", index=False)
        print("Model agreement with experts vs expert-expert agreement (same ROIs; diff CI including 0 = within inter-observer variability):")
        print(IOBS.round(3).to_string(index=False))
else:
    print("No expert reference masks supplied. NO segmentation accuracy (Dice/IoU vs truth) is claimed by this notebook.")

# %% [markdown]
# ## 18. Smallest useful expert-annotation set (active worklist)
# The goal is the fewest annotations that give a usable CI on Dice. The worklist samples cores from three groups:
# * **all** test IHC 2+ (both FISH outcomes) and IHC 3+ cores
# * the IHC-0 cores with the most predicted membranous DAB, which are likely false positives
# * the highest-uncertainty remaining cores
#
# Within each core, 512×512 ROIs (about 235 µm) are chosen by uncertainty, evidence, and at random for an unbiased estimate. Each ROI is exported as a
# PNG together with the model's predicted mask, a CSV in **native** pixel coordinates, and a QuPath-importable GeoJSON.
#
# **Sample-size rationale.** The CI is computed over **cores**, not ROIs, because ROIs within a core are correlated. Assuming a per-core Dice SD
# of about 0.15, n cores gives a 95% CI half-width of about 1.96 × 0.15 / √n. That is ±0.05 at about 35 cores and ±0.04 at about 55 cores.

# %%
WL = OUT / "annotation_worklist"; (WL / "rois").mkdir(exist_ok=True); (WL / "pred_masks").mkdir(exist_ok=True)
t = TSEG.merge(TE_PRED[["image_id", "entropy"]], on="image_id", how="left")
sel = list(t[t.ihc_score.isin([2, 3])].image_id)
sel += list(t[(t.ihc_score == 0) & ~t.image_id.isin(sel)].sort_values("stu_mem_mod", ascending=False).image_id[:12])
sel += list(t[~t.image_id.isin(sel)].sort_values("mean_tta_sd", ascending=False).image_id[:10])
rng = np.random.default_rng(CFG.SEED); wl_rows, feats = [], []
path_of = dict(zip(TE.image_id, TE.path)); S = CFG.SEG_CROP
for iid in tqdm(sel, desc="worklist ROIs"):
    rc = ROIC[ROIC.image_id == iid]
    if rc.empty: continue
    picks = []
    for col in ["tta_sd", "evidence"]:
        c = rc[~rc.index.isin([p.name for p in picks])].sort_values(col, ascending=False)
        if len(c): picks.append(c.iloc[0])
    c = rc[~rc.index.isin([p.name for p in picks])]
    if len(c): picks.append(c.iloc[rng.integers(len(c))])
    work = read_rgb(path_of[iid], CFG.WORK_DS)
    gfile = TEST_OUT / f"{iid}_membrane_grade.png"; grade = cv2.imread(str(gfile), cv2.IMREAD_GRAYSCALE) if gfile.exists() else None
    for k, p in enumerate(picks[:CFG.N_ANNOT_ROIS_PER_CORE]):
        y, x = int(p.y), int(p.x); roi = f"{iid}__roi{k}"
        cv2.imwrite(str(WL / "rois" / f"{roi}.png"), cv2.cvtColor(work[y:y + S, x:x + S], cv2.COLOR_RGB2BGR))
        if grade is not None: cv2.imwrite(str(WL / "pred_masks" / f"{roi}.png"), ((grade[y:y + S, x:x + S] > 0) * 255).astype(np.uint8))
        X0, Y0, side = x * CFG.WORK_DS, y * CFG.WORK_DS, S * CFG.WORK_DS
        why = ["highest uncertainty", "highest evidence", "random (unbiased)"][k] if k < 3 else "extra"
        g = TE.set_index("image_id").loc[iid]
        wl_rows.append(dict(roi_id=roi, image_id=iid, filename=g.filename, ihc_group=g.group, roi_reason=why,
                            x_native=X0, y_native=Y0, w_native=side, h_native=side, work_level_mpp=WORK_MPP))
        feats.append({"_file": g.filename, "type": "Feature", "properties": {"name": roi, "objectType": "annotation", "classification": {"name": "ROI to annotate"}},
                      "geometry": {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + side, Y0], [X0 + side, Y0 + side], [X0, Y0 + side], [X0, Y0]]]}})
WLDF = pd.DataFrame(wl_rows)
# second-annotator subset: about 21 ROIs, stratified by IHC group and ROI reason, one ROI per core (cores are the independent unit)
if len(WLDF):
    one = WLDF.groupby("image_id", group_keys=False).apply(lambda d: d.sample(1, random_state=CFG.SEED))
    IO_SET = one.groupby("ihc_group", group_keys=False).apply(lambda d: d.sample(min(len(d), max(3, round(21 * len(d) / len(one)))),
                                                                                    random_state=CFG.SEED)).head(21)
    WLDF["second_annotator"] = WLDF.roi_id.isin(IO_SET.roi_id)
WLDF.to_csv(WL / "worklist.csv", index=False)
# one GeoJSON per core image: QuPath imports objects into the image that is open, so a combined file would place every core's
# squares on every image
QP = WL / "qupath_geojson"; QP.mkdir(exist_ok=True)
for fname in sorted({f["_file"] for f in feats}):
    fc = [{k: v for k, v in f.items() if k != "_file"} for f in feats if f["_file"] == fname]
    json.dump({"type": "FeatureCollection", "features": fc}, open(QP / f"{Path(fname).stem}.geojson", "w"), indent=1)
(WL / "export_masks_qupath.groovy").write_text(r"""// Export HER2 membrane reference masks: one PNG per worklist ROI, 0.46 um/px, same size as the ROI PNGs (512 x 512).
// QuPath 0.4 or later. Run for the whole project with Automate > Script editor > Run > Run for project.
// Check the first exported mask visually (membrane lines white on black) before exporting everything.
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.regions.RegionRequest

def INITIALS   = ''      // '' for the primary annotator; the second annotator puts initials here, e.g. 'AB'
def DOWNSAMPLE = 2.0     // native 0.23 um/px -> 0.46 um/px (the resolution used by the model)
def OUT_DIR = buildFilePath(PROJECT_BASE_DIR, 'reference_masks' + (INITIALS ? '_' + INITIALS : ''))
mkdirs(OUT_DIR)

def labelServer = new LabeledImageServer.Builder(getCurrentImageData())
    .backgroundLabel(0)
    .downsample(DOWNSAMPLE)
    .addLabel('Membrane', 255)
    .lineThickness(2)
    .multichannelOutput(false)
    .build()

def rois = getAnnotationObjects().findAll { it.getName() != null && it.getName().contains('__roi') }
for (r in rois) {
    def request = RegionRequest.createInstance(labelServer.getPath(), DOWNSAMPLE, r.getROI())
    def name = r.getName() + (INITIALS ? '__' + INITIALS : '') + '.png'
    writeImageRegion(labelServer, request, buildFilePath(OUT_DIR, name))
    print 'wrote ' + name
}
""")
n_cores = WLDF.image_id.nunique() if len(WLDF) else 0
print(f"worklist: {len(WLDF)} ROIs from {n_cores} cores | expected Dice CI half-width ≈ ±{1.96 * 0.15 / np.sqrt(max(n_cores, 1)):.3f}")
(WL / "ANNOTATION_PROTOCOL.md").write_text(f"""# HER2 membrane annotation protocol (gastro-oesophageal adenocarcinoma, Ventana 4B5)
Purpose: reference masks for measuring the accuracy of an automated membrane segmentation. Your masks are the reference standard,
so annotate what you see. Please do not try to match the model.

## Material
* `worklist.csv`: {len(WLDF)} ROIs from {n_cores} test cores. Each ROI is {S}x{S} px at {WORK_MPP:.2f} µm/px (about {S * WORK_MPP:.0f} µm square).
* `qupath_geojson/<core>.geojson`: the ROI squares for each core, in native pixel coordinates of that core's JPEG.
* `rois/`: the same ROIs as PNGs (for a quick look).
* Do **not** open `pred_masks/` before annotating, so that the model does not anchor you.

## What to draw (QuPath 0.4 or later)
1. Create a QuPath project with the listed core JPEGs. For each image, import its own `qupath_geojson/<core>.geojson`
   (File > Import objects, or drag the file onto the image). Each ROI square carries its roi_id as its name.
2. Inside each square, trace **membranous** DAB of **tumour cells only** with the *Polyline* tool (or the brush for thick segments).
   Give each line the class **Membrane**.
   * Count complete, basolateral and lateral (U-shaped) staining. In gastric/GEA cancer, incomplete membranes are scored (Hofmann/Rüschoff).
   * Include weak membranes you would count when scoring: faint, discontinuous lines visible at x40.
   * Exclude cytoplasmic, nuclear and luminal-secretion DAB, stroma, normal glands, and edge or crush artefact.
3. If an ROI contains **no** tumour membrane staining, leave it empty. An empty mask is a valid and important answer.
4. Optional: add an area annotation of class **Tumor** around the tumour epithelium in the ROI.
5. Expected time: about 1 min for an IHC 0 ROI, 3–5 min for a 2+/3+ ROI. Allow roughly 4–6 h in total, split over sessions.

## Second annotator
ROIs with `second_annotator = True` in `worklist.csv` (about 21) should be annotated **independently** by a second pathologist who has not
seen the first annotator's masks. Their masks are saved with an `__<initials>` suffix.

## Export
Run the script `export_masks_qupath.groovy` (Automate > Script editor > Run for project). It writes one PNG per ROI:
`<roi_id>.png`, or `<roi_id>__<initials>.png` when you set INITIALS in the script. Lines are rasterised 2 px wide at 0.46 µm/px.
Put all PNGs in one folder, set `CFG.REF_MASK_DIR` to it, and rerun Section 17.

## Reporting
State the annotators' experience, the tools, that annotation was blind to model output, and the time spent.
""")

# %% [markdown]
# ## 18b. Training convergence and computational cost
# Reviewers of SCIE journals routinely ask whether the models converged and what they cost to run. This cell plots the
# MIL training loss for each fold, and the segmentation students' loss and pseudo-label agreement per epoch.
# It also tabulates the trainable parameters, the measured per-core inference time and the wall-clock time per pipeline section.
# Interrupted-and-resumed sessions are summed.

# %%
def n_params(m, trainable=True): return int(sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable)))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for i, n in enumerate([PROPOSED, "SANC − negative constraint", MAIN_BASELINE, "MaxPool"]):
    L = [l for l in MIL_RES.get(n, {}).get("loss", []) if len(l)]
    if not L: continue
    E = min(map(len, L)); A = np.array([l[:E] for l in L])
    ax[0].plot(np.arange(1, E + 1), A.mean(0), color=PAL[i], label=n); ax[0].fill_between(np.arange(1, E + 1), A.min(0), A.max(0), color=PAL[i], alpha=0.15)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (mean, range over folds)"); ax[0].set_title("MIL")
for i, (n, m) in enumerate(SEG_METRICS.items()):
    h = pd.DataFrame(m["history"])
    ax[1].plot(h.epoch, h.train_loss, "-o", ms=3, color=PAL[i], label=n[:32])
    ax[2].plot(h.epoch, h.agree_Dice, "-o", ms=3, color=PAL[i], label=n[:32])
    ax[2].plot(h.epoch, h.agree_Dice_highconf, ":", lw=1, color=PAL[i])
    b = h[h.epoch == m["final"]["epoch"]].iloc[0]; ax[2].scatter([b.epoch], [b.agree_Dice], s=60, facecolor="none", edgecolor=PAL[i])
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training loss"); ax[1].set_title("Segmentation students")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("pseudo-label agreement, Dice (solid: all pixels; dotted: high-conf)"); ax[2].set_title("Held-out agreement (circle = selected epoch)")
ax[2].legend(fontsize=7)
from matplotlib.ticker import MaxNLocator
for a in ax: a.xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
savefig(fig, "figS2_convergence")

COST = []
use_embedding("emb")
for n, v in VARIANTS.items():
    try: COST.append(dict(component="MIL head", model=n, trainable_params=n_params(build(v))))
    except Exception as e: print("param count failed:", n, e)
for n, m in SEG_MODELS.items(): COST.append(dict(component="segmentation student", model=n, trainable_params=n_params(m)))
try: COST.append(dict(component="CNN baseline (Grad-CAM)", model=CFG.CAM_ARCH, trainable_params=n_params(timm.create_model(CFG.CAM_ARCH, pretrained=False, num_classes=1))))
except Exception as e: print("CAM param count failed:", e)
try:
    _e = timm.create_model(CFG.ENCODER.split(":", 1)[-1], pretrained=False, features_only=True, out_indices=(CFG.ENCODER_OUT_INDEX,))
    COST.append(dict(component="frozen tile encoder (not trained)", model=CFG.ENCODER, trainable_params=n_params(_e, trainable=False))); del _e
except Exception as e: print("encoder param count failed:", e)
COST = pd.DataFrame(COST); atomic_csv(COST, TAB / "compute_parameters.csv", index=False)
if "seg_seconds" in TSEG:
    _t = TSEG.seg_seconds.dropna()
    SEG_TIME = dict(n_cores=int(len(_t)), median_s=float(_t.median()), q25_s=float(_t.quantile(0.25)), q75_s=float(_t.quantile(0.75)),
                    tta=CFG.TTA, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
else: SEG_TIME = {}
save_json(SEG_TIME, TAB / "test_segmentation_time_per_core.json")
try: TIMES = pd.read_csv(OUT / "timings.csv", on_bad_lines="skip")
except Exception: TIMES = pd.DataFrame(columns=["section", "seconds", "session"])
if "session" not in TIMES: TIMES = pd.DataFrame(columns=["section", "seconds", "session"])   # pre-v2.1 log format
_agg = dict(minutes=("seconds", lambda x: x.sum() / 60), sessions=("session", "nunique"))
for c, how in [("ram_gb", "max"), ("peak_gpu_gb", "max"), ("output_gb", "max")]:
    if c in TIMES: _agg[{"ram_gb": "max_RAM_GB", "peak_gpu_gb": "peak_GPU_GB", "output_gb": "output_folder_GB"}[c]] = (c, how)
TIMES = TIMES.groupby("section", sort=False).agg(**_agg).reset_index()
atomic_csv(TIMES, TAB / "compute_section_times.csv", index=False)
print(COST.to_string(index=False)); print("\nwhole-core test segmentation per core:", SEG_TIME); print(TIMES.round(1).to_string(index=False))

# %% [markdown]
# ## 18c. Threshold-free localisation trade-off, publication figures, clean compute table
# **Why.** A single 0.5 threshold can mislead. A model with near-zero scores everywhere (MaxPool) has almost no false positives at 0.5
# simply because it rarely calls any tile positive. The fair comparison sweeps the threshold. For each model it traces the IHC-0 tile
# false-positive rate against the fraction of tiles called positive in IHC-3 cores, then reads off the IHC-3 positivity at a fixed
# false-positive budget (1% and 0.5%). CIs come from a core-level bootstrap (IHC-0 and IHC-3 cores resampled separately).
#
# Tiles in IHC-3 cores are not all tumour, so "IHC-3 tile positivity" is a detection proxy, not a sensitivity. The curve is computed on the
# test set (284 IHC-0 vs 8 IHC-3 cores) and on out-of-fold training cores (about 1,326 vs 114), which give much tighter intervals.
#
# This cell also redraws Fig 8, Fig 10, Fig 12 and Fig S2 with readable labels and white backgrounds, and writes a de-duplicated
# compute table.

# %%
def _tile_sets(res, split):
    """Per-core tile probabilities for IHC-0 and IHC-3 cores (fold-averaged on test; OOF fold model on train)."""
    if split == "test":
        sc = per_fold_tile_scores(res, "inst")
        if sc is None: return None
        P = [sig(x.mean(0)) for x in sc]
        return [P[b] for b in neg0], [P[b] for b in ihc3]
    oof = res.get("oof")
    if oof is None or oof[0] is None or oof[0].get("inst") is None: return None
    n0 = np.where(TR.ihc_score.values == 0)[0]; n3 = np.where(TR.ihc_score.values == 3)[0]
    return [sig(oof[j]["inst"]) for j in n0], [sig(oof[j]["inst"]) for j in n3]

def _pos_at_fp(neg_cores, pos_cores, fp):
    t = np.quantile(np.concatenate(neg_cores), 1 - fp)                 # threshold giving the target pooled IHC-0 tile FP rate
    return float(np.mean(np.concatenate(pos_cores) > t))

OPC_METHODS = [m for m in [PROPOSED, "SANC − negative constraint", "SANC − stain features", "MaxPool", "CLAM-SB", "MeanPool",
                           "Grad-CAM CNN (per-image normalised CAM)"] if m in {**MIL_RES, **CAM_RES}]
OPC_ROWS, OPC_CURVES = [], {}
fp_grid = np.logspace(-4, np.log10(0.3), 60)
for split in ("test", "oof"):
    for m in OPC_METHODS:
        ts = _tile_sets({**MIL_RES, **CAM_RES}[m], split)
        if ts is None: continue
        neg, pos = ts
        OPC_CURVES[(m, split)] = [_pos_at_fp(neg, pos, f) for f in fp_grid]
        rb = np.random.default_rng(0); row = dict(method=m, split=split, n_IHC0_cores=len(neg), n_IHC3_cores=len(pos))
        for fp in (0.01, 0.005):
            v = _pos_at_fp(neg, pos, fp)
            bs = [_pos_at_fp([neg[i] for i in rb.integers(0, len(neg), len(neg))], [pos[i] for i in rb.integers(0, len(pos), len(pos))], fp)
                  for _ in range(500)]
            row.update({f"IHC3_tile_pos@FP{fp:g}": v, f"lo@FP{fp:g}": float(np.quantile(bs, 0.025)), f"hi@FP{fp:g}": float(np.quantile(bs, 0.975))})
        y = np.r_[np.ones(sum(map(len, pos))), np.zeros(sum(map(len, neg)))]
        row["tile_AUROC_IHC3_vs_IHC0"] = float(roc_auc_score(y, np.r_[np.concatenate(pos), np.concatenate(neg)]))
        OPC_ROWS.append(row)
OPC = pd.DataFrame(OPC_ROWS); atomic_csv(OPC, TAB / "localisation_operating_points.csv", index=False)
print("IHC-3 tile positivity at a fixed IHC-0 tile false-positive budget (core-bootstrap 95% CI):")
print(OPC.round(3).to_string(index=False))

LABEL = {PROPOSED: "SANC-MIL (proposed)", "SANC − negative constraint": "SANC without negative constraint",
         "SANC − stain features": "SANC without stain features", "MaxPool": "MaxPool", "CLAM-SB": "CLAM-SB", "MeanPool": "MeanPool",
         "Grad-CAM CNN (per-image normalised CAM)": "Grad-CAM CNN"}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
def _n(split, col):
    v = OPC[OPC.split == split][col]; return int(v.iloc[0]) if len(v) else 0
for j, (split, ttl) in enumerate([("test", f"Test set ({_n('test', 'n_IHC0_cores')} IHC 0 vs {_n('test', 'n_IHC3_cores')} IHC 3 cores)"),
                                  ("oof", f"Out-of-fold training cores ({_n('oof', 'n_IHC0_cores')} vs {_n('oof', 'n_IHC3_cores')})")]):
    for i, m in enumerate(OPC_METHODS):
        if (m, split) not in OPC_CURVES: continue
        ax[j].plot(fp_grid * 100, np.array(OPC_CURVES[(m, split)]) * 100, color=PAL[i % len(PAL)], lw=2.2 if m == PROPOSED else 1.4,
                   ls="-" if m != "SANC − negative constraint" else "--", label=LABEL.get(m, m))
    ax[j].set_xscale("log"); ax[j].axvline(1, color="#888", lw=0.8, ls=":"); ax[j].set_title(ttl, fontsize=9)
    ax[j].set_xlabel("IHC 0 tiles called positive (%, log scale)")
ax[0].set_ylabel("IHC 3 tiles called positive (%)")
_h, _l = ax[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Localisation trade-off across all thresholds (dotted line: 1% false-positive budget)", fontsize=10)
savefig(fig, "fig11b_localisation_operating_curve")

# ---------- readable feature names for figures and tables ----------
def pretty_feature(f):
    src = "student" if f.startswith("stu_") else ("rule" if f.startswith("rule_") else "")
    g = re.sub(r"^(stu|rule)_", "", f)
    names = {"mem_weak_frac": "membranous DAB area, ≥weak", "mem_mod_frac": "membranous DAB area, ≥moderate",
             "mem_strong_frac": "membranous DAB area, ≥strong", "cell_cov>=0.3": "cells ≥30% membrane-covered",
             "cell_cov>=0.5": "cells ≥50% membrane-covered", "cell_cov>=0.7": "cells ≥70% membrane-covered",
             "cell_cov>=0.5&mod": "cells ≥50% covered at ≥moderate DAB", "cell_cov>=0.5&strong": "cells ≥50% covered at ≥strong DAB",
             "cell_median_cov_pos": "median coverage of stained cells", "cell_n_cells": "number of cells"}
    return f"{names.get(g, g)} ({src})" if src else names.get(g, g)

# ---------- Fig 10 redrawn ----------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax_, f in zip(axes, ["stu_mem_mod_frac", "stu_cell_cov>=0.5", "stu_cell_cov>=0.5&mod", "rule_mem_mod_frac"]):
    for k, g in enumerate(GROUP_ORDER):
        v = cf_te[cf_te.group == g][f].dropna().values if f in cf_te else []
        mk = "o"
        if len(v) == 0: v = cf_tr[cf_tr.group == g][f].dropna().values; mk = "x"
        ax_.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v, s=9, color=GROUP_COL[g], alpha=0.6, marker=mk)
        if len(v): ax_.hlines(np.median(v), k - 0.3, k + 0.3, color="#0b0b0b", lw=1.5)
    ax_.set_xticks(range(len(GROUP_ORDER))); ax_.set_xticklabels(["0", "1+", "2+ FISH−", "2+ FISH+", "3+"], fontsize=7)
    ax_.set_title(pretty_feature(f), fontsize=8)
axes[0].set_ylabel("fraction per core")
fig.text(0.5, -0.02, "Test cores (●); IHC 1+ shown from training cores (×) because the test set has none. Bars = medians.", ha="center", fontsize=7.5)
savefig(fig, "fig10_clinical_consistency")

# ---------- Fig 12 redrawn ----------
d = H3[H3.comparison == "3+ vs 2+"].pivot_table(index="feature", columns="split", values=["AUROC", "lo", "hi"])
if ("AUROC", "test") in d.columns:
    d = d.sort_values(("AUROC", "train"), ascending=False).head(12)
    fig, ax_ = plt.subplots(figsize=(7.5, 0.36 * len(d) + 1.3)); yy = np.arange(len(d))
    for off, sp, col, lab in [(-0.15, "train", PAL[0], "training cores (73 vs 114)"), (0.15, "test", PAL[7], "test cores (15 vs 8)")]:
        ax_.errorbar(d[("AUROC", sp)], yy + off, xerr=[d[("AUROC", sp)] - d[("lo", sp)], d[("hi", sp)] - d[("AUROC", sp)]],
                     fmt="o", ms=4, color=col, label=lab, capsize=2)
    ax_.axvline(0.5, color="#888", ls="--", lw=1); ax_.set_yticks(yy); ax_.set_yticklabels([pretty_feature(f) for f in d.index], fontsize=7)
    ax_.invert_yaxis(); ax_.set_xlabel("AUROC, IHC 3+ vs IHC 2+ (95% stratified bootstrap CI)"); ax_.legend(fontsize=7, loc="lower left")
    savefig(fig, "fig12_h3_3plus_vs_2plus")

# ---------- Fig 8 redrawn: white background, sequential maps that start at white, colour bars ----------
def heatmap_figure_pub(n_per=1, seed=CFG.SEED):
    r_ = np.random.default_rng(seed); picks = []
    for g in ["0", "2-", "2+", "3"]:
        s_ = TE[TE.group == g]
        if len(s_): picks += [s_.index[i] for i in r_.choice(len(s_), min(n_per, len(s_)), replace=False)]
    inst_s = per_fold_tile_scores(PROPOSED, "inst"); att_a = per_fold_tile_scores(MAIN_BASELINE, "attn")
    cam_n = per_fold_tile_scores(CAM_RES["Grad-CAM CNN (per-image normalised CAM)"], "inst") if CAM_RES else None
    cols = [("Core", None, None), ("DAB optical density", "YlOrBr", (0, CALIB["t_strong"] * 1.2)),
            ("Membranous DAB ≥ moderate\n(fraction per pixel block)", "Reds", None), ("Grad-CAM\n(per-image normalised)", "viridis", (0, 1)),
            ("ABMIL attention\n(scaled to max)", "viridis", (0, 1)), ("SANC-MIL P(positive tile)", "RdBu_r", (0, 1)), ("SANC-MIL uncertainty\n(SD over 5 folds)", "Greys", (0, 0.35))]
    fig, axes_all = plt.subplots(len(picks) + 1, len(cols), figsize=(2.3 * len(cols), 2.35 * len(picks) + 0.7),
                                 gridspec_kw={"height_ratios": [1] * len(picks) + [0.05]})
    axes, caxes = axes_all[:-1], axes_all[-1]                                  # last row holds the colour bars
    ims = [None] * len(cols)
    for r, b in enumerate(picks):
        iid = TE.image_id[b]; mpz = np.load(MAP_DIR / f"{iid}.npz"); th = cv2.cvtColor(cv2.imread(str(THUMB_DIR / f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
        shp = mpz["dab"].shape
        tmask = cv2.resize(mpz["tissue"].astype(np.uint8), (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST) > 0
        inside = lambda a: np.where(tmask, a.astype(np.float32), np.nan)
        pr = sig(inst_s[b]); att = att_a[b].mean(0); att = att / (att.max() + 1e-9)
        camv = tile_grid_map(iid, sig(cam_n[b]).mean(0), shp) if cam_n is not None else np.full(shp, np.nan, np.float32)
        mm = inside(mpz["mem_mod"])
        data = [th, inside(mpz["dab"]), mm, camv, tile_grid_map(iid, att, shp), tile_grid_map(iid, pr.mean(0), shp), tile_grid_map(iid, pr.std(0), shp)]
        for c, ((title, cmap, lim), img) in enumerate(zip(cols, data)):
            a = axes[r, c]; a.axis("off")
            if cmap is None: a.imshow(img)
            else:
                cm_ = plt.get_cmap(cmap).copy(); cm_.set_bad("white")
                vmin, vmax = lim if lim else (0, max(0.05, float(np.nanpercentile(img, 99.5)) if np.isfinite(img).any() else 0.05))
                ims[c] = a.imshow(cv2.resize(img, (th.shape[1], th.shape[0]), interpolation=cv2.INTER_NEAREST), cmap=cm_, vmin=vmin, vmax=vmax)
            if r == 0: a.set_title(title, fontsize=8)
        lab = {"0": "IHC 0", "2-": "IHC 2+, FISH−", "2+": "IHC 2+, FISH+", "3": "IHC 3+"}[TE.group[b]]
        axes[r, 0].text(5, 25, f"{lab}   P(HER2+)={pu['pm'][b]:.2f}", color="k", fontsize=7, bbox=dict(fc="w", ec="none", alpha=0.85))
    for c, im in enumerate(ims):
        if im is not None:
            cb = fig.colorbar(im, cax=caxes[c], orientation="horizontal"); cb.ax.tick_params(labelsize=6)
        else: caxes[c].axis("off")
    savefig(fig, "fig08_localization_maps")
heatmap_figure_pub()

# ---------- Fig S2 label fix ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for i, n in enumerate([PROPOSED, "SANC − negative constraint", MAIN_BASELINE, "MaxPool"]):
    L = [l for l in MIL_RES.get(n, {}).get("loss", []) if len(l)]
    if not L: continue
    E = min(map(len, L)); A = np.array([l[:E] for l in L])
    ax[0].plot(np.arange(1, E + 1), A.mean(0), color=PAL[i], label=LABEL.get(n, n)); ax[0].fill_between(np.arange(1, E + 1), A.min(0), A.max(0), color=PAL[i], alpha=0.15)
for i, (n, m_) in enumerate(SEG_METRICS.items()):
    h = pd.DataFrame(m_["history"])
    ax[1].plot(h.epoch, h.train_loss, "-o", ms=3, color=PAL[i], label=n.replace(" (proposed)", ""))
    ax[2].plot(h.epoch, h.agree_Dice, "-o", ms=3, color=PAL[i]); ax[2].plot(h.epoch, h.agree_Dice_highconf, ":", lw=1, color=PAL[i])
    b_ = h[h.epoch == m_["final"]["epoch"]].iloc[0]; ax[2].scatter([b_.epoch], [b_.agree_Dice], s=60, facecolor="none", edgecolor=PAL[i])
ax[0].set(xlabel="epoch", ylabel="training loss", title="MIL (mean, range over 5 folds)"); ax[0].legend(fontsize=7)
ax[1].set(xlabel="epoch", ylabel="training loss", title="Segmentation students"); ax[1].legend(fontsize=6.5)
ax[2].set(xlabel="epoch", ylabel="agreement with pseudo-labels (Dice)", title="Held-out agreement (solid: all pixels;\ndotted: high-confidence; circle: kept epoch)")
from matplotlib.ticker import MaxNLocator
for a in ax: a.xaxis.set_major_locator(MaxNLocator(integer=True))
savefig(fig, "figS2_convergence")

# ---------- clean compute table: a section logged twice in one session = a re-executed cell, not new work ----------
try:
    _T = pd.read_csv(OUT / "timings.csv", on_bad_lines="skip")
    if "session" in _T:
        _T["_n"] = _T.groupby(["section", "session"]).cumcount()
        dup = _T[_T._n > 0]
        clean = _T[_T._n == 0].groupby("section", sort=False).agg(minutes=("seconds", lambda x: x.sum() / 60), sessions=("session", "nunique"),
                                                                  peak_GPU_GB=("peak_gpu_gb", "max"), max_RAM_GB=("ram_gb", "max")).reset_index()
        clean["note"] = np.where(clean.sessions > 1, "summed over sessions (a resumed session adds only the unfinished work)", "")
        atomic_csv(clean, TAB / "compute_section_times_clean.csv", index=False)
        print("\nCompute per section (re-executed cells removed):\n", clean.round(1).to_string(index=False))
        if len(dup): print("removed duplicate entries:", list(dup.section))
except Exception as e:
    print("compute table not rebuilt:", e)
section_time("figures + operating curve")

# %% [markdown]
# ## 19. Results summary and claims ledger
# The summary below is generated automatically from the tables. The ledger separates what the evidence **supports** from what remains **unverified**.

# %%
def md(df):
    try: return df.to_markdown(index=False)
    except Exception: return "```\n" + df.to_string(index=False) + "\n```"
def fmt_ci(r, k): return f"{r[k]:.3f} (95% CI {r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})"
pr = CLS[CLS.method == PROPOSED].iloc[0]; bl = CLS[CLS.method == MAIN_BASELINE].iloc[0]; ir = CLS[CLS.method == "Stain intensity rule (Pisula 2023)"].iloc[0]
d_ab = DELTA[DELTA.comparison.str.endswith(MAIN_BASELINE)].iloc[0]
locp = LOC.set_index("method")
# ---- evidence-gated claim wording: a hypothesis is called "supported" only when the pre-stated criterion is met ----
_h1 = H1[(H1.A == PROPOSED) & (H1.B == "SANC − negative constraint")] if len(H1) else pd.DataFrame()
H1_OK = bool(len(_h1)) and bool((_h1.hi < 0).all())
H1_TXT = ("SUPPORTED: in every seed and on both the test set and OOF train, the CI of the per-core FP difference lies below 0 "
          f"(mean diff {_h1['diff'].mean():+.3f})." if H1_OK else
          "NOT consistently supported: at least one seed/split has a CI including 0 — report as a trend, see table.") if len(_h1) else "not computed"
_h3 = H3[H3.comparison == "3+ vs 2+"] if len(H3) else pd.DataFrame()
if len(_h3) and (_h3.split == "test").any():
    _best = _h3[_h3.split == "train"].sort_values("AUROC", ascending=False).feature.iloc[0]       # chosen on TRAIN, evaluated on test
    _bt = _h3[(_h3.split == "test") & (_h3.feature == _best)]
    H3_TXT = (f"train-selected feature `{_best}`: test AUROC {_bt.AUROC.iloc[0]:.3f} (95% CI {_bt.lo.iloc[0]:.3f}–{_bt.hi.iloc[0]:.3f}, "
              f"n={_bt.n_pos.iloc[0]} vs {_bt.n_neg.iloc[0]}) — " + ("SUPPORTED (CI excludes 0.5)." if _bt.lo.iloc[0] > 0.5 else "NOT supported (CI includes 0.5).")) if len(_bt) else "not computable on test"
else: H3_TXT = "not computable"
lines = [
    "# Results summary (auto-generated)", "",
    f"*Run mode:* `{RUN_MODE}` | encoder `{CFG.ENCODER}` (pretrained={ENCODER_PRETRAINED}) | train cores {len(TR)} | test cores {len(TE)} "
    f"({int(ystat_te.sum())} HER2+) | test groups: " + ", ".join(f"{g}: {int((TE.group == g).sum())}" for g in GROUP_ORDER), "",
    "## Classification (temporally independent test set)", "", md(CLS_PRETTY), "",
    f"Paired bootstrap ΔAUROC, {PROPOSED} vs {MAIN_BASELINE}: {d_ab.dAUC:+.3f} (95% CI {d_ab.lo:+.3f} to {d_ab.hi:+.3f}, p={d_ab.p_boot:.3f}).", "",
    "All paired comparisons with the proposed model (bootstrap p, Holm-corrected):", "", md(DELTA.round(4)), "",
    "Leakage sensitivity (OOF AUROC, random vs proxy-grouped CV):", "", md(LEAK.round(4)), "",
    "Abstention / equivocal routing:", "", "```", json.dumps(ab, indent=1, default=str), "```", "",
    "## Localisation (grounded checks + proxies)", "", md(LOC.round(3)), "",
    f"Variability over {CFG.N_SEEDS} repeated CV seeds (mean, SD):", "", "```", SEED_SUMMARY.to_string(), "```", "",
    "### H1: paired per-core tests of IHC-0 false-positive tile fraction (A − B < 0 favours A)", "", md(H1.round(4)), "",
    "### Threshold-free trade-off: IHC-3 tile positivity at a fixed IHC-0 tile false-positive budget", "", md(OPC.round(3)), "",
    "## Segmentation student (agreement with PSEUDO-labels on held-out cores — not accuracy)", "", md(SEGT.round(3)), "",
    "Test-set proxies:", "", md(SEG_PROXY.round(4)), "",
    "## Clinical consistency of segmentation-derived features", "", md(CLIN.round(3)), "", md(TREND.round(3)), "",
    "### H3: single-feature separation of 3+ vs 2+ and of 2+/FISH+ vs 2+/FISH− (direction fixed on train; top 5 per block)", "",
    md(H3.groupby(["comparison", "split"], sort=False).head(5).round(3)) if len(H3) else "(not computable)", "",
    "## Computational cost", "", md(COST), "", f"Whole-core test segmentation: {SEG_TIME}", "", md(TIMES.round(1)), "",
    "## Claims ledger", "",
    "**Supported by this run (with the stated uncertainty):**",
    f"- Image-level HER2 status on the independent test set: SANC-MIL AUROC {fmt_ci(pr, 'AUROC')}; ABMIL {fmt_ci(bl, 'AUROC')}; stain-intensity rule {fmt_ci(ir, 'AUROC')}.",
    "- Instance positivity on IHC-0 test cores (label-grounded specificity), seed mean ± SD: " + "; ".join(
        f"{m}: FP@0.5 {SEED_SUMMARY.loc[m, ('IHC0_tile_FP_rate@0.5', 'mean')]:.3f} ± {SEED_SUMMARY.loc[m, ('IHC0_tile_FP_rate@0.5', 'std')]:.3f}"
        for m in SEED_SUMMARY.index if ("IHC0_tile_FP_rate@0.5", "mean") in SEED_SUMMARY.columns and not np.isnan(SEED_SUMMARY.loc[m, ("IHC0_tile_FP_rate@0.5", "mean")])) + ".",
    "- Grad-CAM baseline (per-image normalised, as in the original draft): IHC-0 tile FP@0.5 = "
    + (f"{locp.loc['Grad-CAM CNN (per-image normalised CAM)', 'IHC0_tile_FP_rate@0.5']:.3f}" if "Grad-CAM CNN (per-image normalised CAM)" in locp.index else "n/a") + ".",
    f"- ISH routing (threshold tuned on OOF for {CFG.ISH_ROUTE_TARGET_SENS:.0%} sensitivity): test sensitivity for IHC 2 = {ab['ISH_routing_sensitivity_for_true_IHC2']:.2f}, routing rate in non-IHC 2 = {ab['ISH_routing_rate_in_non_IHC2']:.2f}.",
    f"- H1 (negative-bag constraint reduces IHC-0 false-positive tiles): {H1_TXT}",
    f"- H3 (membrane/cell features separate IHC 3+ from 2+): {H3_TXT}",
    "- Classification differences between MIL variants: see Holm-corrected table; a difference is claimed only where p_holm < 0.05.",
    "- Monotonic association of membrane-derived features with IHC score (see Spearman ρ / Kruskal–Wallis table).",
    "- QC and leakage analyses: see qc_summary.json and leakage_sensitivity.csv.", "",
    "**NOT established (do not claim):**",
    "- Pixel-level segmentation accuracy of membranes, tumour regions or cells: no expert masks were available"
    + (" — EXCEPT the reference-mask results in reference_mask_metrics_summary.csv." if CFG.REF_MASK_DIR and Path(CFG.REF_MASK_DIR).exists() else "."),
    "- That 'pseudo-label agreement' Dice equals accuracy — it measures reproduction of the training signal.",
    "- That attention / evidence maps mark the cells a pathologist would score — they mark what drives the model.",
    "- Any IHC 1+ test performance (the test set has no 1+ cores); IHC 2+ results rest on n≈15 cores.",
    "- That the model can replace FISH for IHC 2+: equivocal cores are routed to ISH by design.",
    "- Generalisation across institutions/scanners: single centre, single scanner here.", "",
    f"Next step for validated segmentation: annotate `annotation_worklist/` ({len(WLDF)} ROIs, {n_cores} cores; expected Dice CI ±{1.96 * 0.15 / np.sqrt(max(n_cores, 1)):.2f}).",
]
(OUT / "RESULTS_SUMMARY.md").write_text("\n".join(lines))
section_time("report")
env = dict(version="v2.1", run_mode=RUN_MODE, torch=torch.__version__, timm=timm.__version__, numpy=np.__version__, sklearn=__import__("sklearn").__version__,
           skimage=skimage.__version__, cv2=cv2.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           encoder=CFG.ENCODER, encoder_pretrained=ENCODER_PRETRAINED, encoder2=CFG.ENCODER2, seed=CFG.SEED,
           cfg={k: v for k, v in vars(CFG).items() if not k.startswith("_") and isinstance(v, (int, float, str, tuple, bool, type(None)))})
save_json(env, OUT / "environment_and_config.json")
print("\n".join(lines[:12]))
print(f"\nAll outputs in {OUT}: tables/, figures/ (PNG 300 dpi + PDF), test_outputs/, annotation_worklist/, models/, RESULTS_SUMMARY.md")
