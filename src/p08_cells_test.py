# %% [markdown]
# ## 14. Cell-level membrane morphometry and core-level features (same protocol for train and test)
# Each core contributes `FEAT_CROPS_PER_CORE` interior crops. When cluster naming is available, the crops come from tumour clusters only.
# For each crop:
# 1. **Nuclei** are haematoxylin-**dominant** blobs (H ≥ `t_h` and H ≥ DAB), split by a distance-transform watershed.
# 2. **Cells (v2, membrane-guided)** come from a marker watershed. The nuclei are the seeds, the smoothed membrane map is the elevation, and growth is limited to `CELL_MAX_UM`. Cell outlines follow stained membranes where they exist.
# 3. **Membrane coverage** is the fraction of each territory boundary that lies on predicted membranous DAB, together with the mean DAB OD on that
#    covered boundary.
#
# Coverage is reported in bins (≥0.3, 0.5, 0.7). These bins describe a distribution. They are **not** a re-implementation of "complete/basolateral"
# scoring. Territories are approximations, and without a tumour mask stromal cells are included.
#
# The same features are computed from the **rule-based** membrane map and from the **student**. This lets the analysis test whether distillation adds
# anything downstream. Only membrane-derived features are used for the clinical classifier. The evidence head saw the labels, so it is excluded.

# %%
from multiprocessing import Pool
CAL_T = dict(t_h=CALIB["t_h"], t_weak=CALIB["t_weak"], t_mod=CALIB["t_mod"], t_strong=CALIB["t_strong"])
def cell_analysis(args):
    """Membrane-guided cell morphometry (v2). Nuclei (haematoxylin-dominant) are watershed seeds; the membrane map is the
    barrier, so cell outlines follow stained membranes where present and stay compact where absent. Coverage = fraction
    of each cell's boundary that lies on membranous DAB (complete, basolateral and lateral patterns give high/partial values)."""
    Hc, Dc, tis, mems = args
    hs, ds = ndi.gaussian_filter(Hc, 1.0), ndi.gaussian_filter(Dc, 1.0)
    nucm = ndi.binary_opening((hs >= CAL_T["t_h"]) & (hs >= ds) & tis, iterations=1)
    if nucm.sum() < 20:
        return [(np.zeros(0), np.zeros(0)) for _ in mems]
    dist = ndi.distance_transform_edt(nucm)
    lab0, _ = ndi.label(nucm)
    pk = peak_local_max(dist, min_distance=max(2, int(um2px(CFG.CELL_NUC_MIN_UM))), labels=lab0, exclude_border=False)
    mk = np.zeros_like(lab0); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    nuc = watershed(-dist, mk, mask=nucm)
    area = np.bincount(nuc.ravel()); small = np.where(area < math.pi * um2px(1.5) ** 2)[0]
    nuc[np.isin(nuc, small)] = 0
    territory = (expand_labels(nuc, distance=um2px(CFG.CELL_MAX_UM)) > 0) & tis
    out = []
    for mem in mems:
        elev = ndi.gaussian_filter(mem.astype(np.float32), 1.0)
        cells = watershed(elev, markers=nuc, mask=territory, compactness=0.02)
        bnd = find_boundaries(cells, mode="inner") & (cells > 0) & (nuc == 0)
        n = cells.max() + 1; tot = np.bincount(cells[bnd], minlength=n)
        md = ndi.binary_dilation(mem, iterations=2); hit = bnd & md
        cov = np.bincount(cells[hit], minlength=n); dsum = np.bincount(cells[hit], weights=Dc[hit], minlength=n)
        ok = tot >= 8; ok[0] = False
        out.append((cov[ok] / tot[ok], np.where(cov[ok] > 0, dsum[ok] / np.maximum(cov[ok], 1), 0.0)))
    return out

def summarize_cells(cov, dab, prefix):
    n = len(cov); f = lambda m: float(m.mean()) if n else 0.0
    return {f"{prefix}n_cells": n, f"{prefix}cov>=0.3": f(cov >= 0.3), f"{prefix}cov>=0.5": f(cov >= 0.5), f"{prefix}cov>=0.7": f(cov >= 0.7),
            f"{prefix}cov>=0.5&mod": f((cov >= 0.5) & (dab >= CAL_T["t_mod"])), f"{prefix}cov>=0.5&strong": f((cov >= 0.5) & (dab >= CAL_T["t_strong"])),
            f"{prefix}median_cov_pos": float(np.median(cov[cov > 0])) if (cov > 0).any() else 0.0}

def remove_small(mask_np, min_px):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]

@torch.no_grad()
def seg_predict(model, x, tta=1):
    """x: (B,3,H,W) float [0,1]. Returns p_mem, p_ev, sd_mem (B,H,W)."""
    pm, pe = [], []
    tfs = [(lambda z: z, lambda z: z), (lambda z: z.flip(-1), lambda z: z.flip(-1)), (lambda z: z.flip(-2), lambda z: z.flip(-2)),
           (lambda z: torch.rot90(z, 1, (-2, -1)), lambda z: torch.rot90(z, -1, (-2, -1)))][:max(1, tta)]
    for f, finv in tfs:
        with autocast(): lm, le = model(f(x))
        pm.append(finv(torch.sigmoid(lm.float()))); pe.append(finv(torch.sigmoid(le.float())))
    pm = torch.stack(pm); pe = torch.stack(pe)
    return pm.mean(0), pe.mean(0), pm.std(0) if len(tfs) > 1 else torch.zeros_like(pm[0])

FEAT_PATH = CACHE / "core_features.csv"
SEG_P = SEG_MODELS[SEG_PROPOSED]
if FEAT_PATH.exists():
    COREF = pd.read_csv(FEAT_PATH)
else:
    S = CFG.SEG_CROP
    flog = ProgressLog("core_features")                   # restart skips cores already measured
    both = pd.concat([TR, TE], ignore_index=True)
    both = both[[i not in flog for i in both.image_id]].reset_index(drop=True)
    print(f"core & cell features: {len(flog)} cores done earlier, {len(both)} to go")
    pool = Pool(CFG.NUM_WORKERS) if CFG.NUM_WORKERS > 1 and len(both) else None
    dl = torch.utils.data.DataLoader(DecodeDS(both), batch_size=1, num_workers=CFG.NUM_WORKERS, collate_fn=lambda b: b[0])
    for i, work in tqdm(dl, total=len(both), desc="core & cell features"):
        r = both.iloc[i]; iid = r.image_id
        if work is None or work.shape[0] < S or work.shape[1] < S:
            flog.add(iid, dict(_skip=True)); continue
        rng = np.random.default_rng(_img_seed(iid) + 7)
        mz = np.load(MAP_DIR / f"{iid}.npz")
        mp = ENGINE.maps(work, mz["I0"], mz["tissue"], mz["rim"])
        tis = mp["tissue"][0, 0]; interior = tis & ~mp["rim"][0, 0]
        b = BAGS[iid]; T = CFG.TILE
        cand = b["coords"] if TUMOUR_CLUSTERS is None else b["coords"][np.isin(b["tc"], list(TUMOUR_CLUSTERS))]
        if len(cand) == 0: cand = b["coords"]
        sel = cand[rng.choice(len(cand), min(CFG.FEAT_CROPS_PER_CORE, len(cand)), replace=False)]
        H, W = work.shape[:2]
        boxes = [(int(np.clip(y + T // 2 - S // 2, 0, H - S)), int(np.clip(x + T // 2 - S // 2, 0, W - S))) for y, x in sel]
        xb = torch.stack([torch.from_numpy(work[y:y + S, x:x + S]) for y, x in boxes]).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        pmem, _, _ = seg_predict(SEG_P, xb, tta=1)
        args, acc = [], defaultdict(float)
        for j, (y, x) in enumerate(boxes):
            sl = (slice(y, y + S), slice(x, x + S))
            ti = interior[sl].cpu().numpy(); d = mp["DAB"][0, 0][sl].cpu().numpy(); hh = mp["H"][0, 0][sl].cpu().numpy()
            stu = remove_small((pmem[j].cpu().numpy() >= 0.5) & ti, CFG.MIN_MEM_OBJ_PX)
            rule = mp["mem_weak"][0, 0][sl].cpu().numpy() & ti
            acc["tissue_px"] += ti.sum()
            for nm, mk in [("stu", stu), ("rule", rule)]:
                for k in ("weak", "mod", "strong"): acc[f"{nm}_mem_{k}"] += (mk & (d >= CALIB[f"t_{k}"])).sum()
            args.append((hh, d, ti, [stu, rule]))
        res = None
        if pool is not None:
            try:
                res = pool.map_async(cell_analysis, args).get(timeout=600)   # never hang: fall back to serial
            except Exception as e:
                print(f"!! parallel cell analysis failed ({type(e).__name__}); continuing single-process"); pool.terminate(); pool = None
        if res is None: res = [cell_analysis(a) for a in args]
        rec = dict(image_id=iid, split=r.split, group=r.group, ihc_score=r.ihc_score, her2_status=r.her2_status)
        for k, v in acc.items():
            if k != "tissue_px": rec[k + "_frac"] = v / max(acc["tissue_px"], 1)
        for mi, nm in enumerate(["stu", "rule"]):
            cov = np.concatenate([rr[mi][0] for rr in res]); dd = np.concatenate([rr[mi][1] for rr in res])
            rec.update(summarize_cells(cov, dd, f"{nm}_cell_"))
        flog.add(iid, rec); del mp
    if pool: pool.close()
    COREF = pd.DataFrame([{k: v for k, v in rec.items() if k != "_key"} for rec in flog.records() if not rec.get("_skip")])
    atomic_csv(COREF, FEAT_PATH, index=False)
section_time("core & cell features")

# %% [markdown]
# ## 15. Whole-core test inference: membrane masks, intensity grades, evidence maps, and uncertainty
# The student runs as a sliding window with Hann blending and `TTA` flips/rotations. It outputs:
# * **membrane** = P ≥ 0.5 inside tissue, with small objects removed and no morphological closing, so thin structures survive
# * an **intensity grade** for each membrane pixel, *measured* from absolute DAB OD rather than predicted
# * an **evidence region** map
# * **uncertainty**, as the SD across TTA
#
# The stain-perturbation self-consistency check re-segments a subset with a fixed ±5% stain shift.

# %%
TEST_OUT = OUT / "test_outputs"
@torch.no_grad()
def segment_core(model, work, tissue, tta):
    S = CFG.SEG_CROP; stride = S - S // 4; H, W = work.shape[:2]
    ys, xs = torch.nonzero(tissue, as_tuple=True)
    if len(ys) == 0: z = np.zeros((H, W), np.float32); return z, z, z, []
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    gy = list(range(max(0, y0 - S // 4), max(1, min(H - S, y1) + 1), stride)) or [0]
    gx = list(range(max(0, x0 - S // 4), max(1, min(W - S, x1) + 1), stride)) or [0]
    if gy[-1] + S < min(H, y1 + 1): gy.append(H - S)
    if gx[-1] + S < min(W, x1 + 1): gx.append(W - S)
    win = torch.hann_window(S, periodic=False, device=DEVICE); win = (0.1 + 0.9 * win[:, None] * win[None, :])
    acc = torch.zeros(3, H, W, device=DEVICE); wsum = torch.zeros(H, W, device=DEVICE)
    xg = torch.from_numpy(work).to(DEVICE).permute(2, 0, 1)
    boxes = [(y, x) for y in gy for x in gx if tissue[y:y + S, x:x + S].float().mean() > 0.02]
    for b in range(0, len(boxes), CFG.SEG_BS):
        bb = boxes[b:b + CFG.SEG_BS]
        xb = torch.stack([xg[:, y:y + S, x:x + S] for y, x in bb]).float() / 255
        pm, pe, sd = gpu_safe(lambda t: seg_predict(model, t, tta), xb)
        for j, (y, x) in enumerate(bb):
            acc[0, y:y + S, x:x + S] += pm[j] * win; acc[1, y:y + S, x:x + S] += pe[j] * win; acc[2, y:y + S, x:x + S] += sd[j] * win
            wsum[y:y + S, x:x + S] += win
    acc = acc / wsum.clamp(min=1e-6)
    return acc[0].cpu().numpy(), acc[1].cpu().numpy(), acc[2].cpu().numpy(), boxes

def dice_np(a, b):
    s = a.sum() + b.sum(); return float(2 * (a & b).sum() / s) if s else 1.0

TEST_SEG_PATH = CACHE / "test_seg_summary.csv"; ROI_PATH = CACHE / "test_roi_candidates.csv"
if TEST_SEG_PATH.exists():
    TSEG = pd.read_csv(TEST_SEG_PATH); ROIC = pd.read_csv(ROI_PATH)
else:
    perturb_ids = set(TE.image_id.sample(min(40, len(TE)), random_state=CFG.SEED))   # fixed set, independent of restarts
    slog = ProgressLog("test_seg")
    todo = TE[[i not in slog for i in TE.image_id]].reset_index(drop=True)
    print(f"test segmentation: {len(slog)} cores done earlier, {len(todo)} to go")
    dl = torch.utils.data.DataLoader(DecodeDS(todo), batch_size=1, num_workers=CFG.NUM_WORKERS, collate_fn=lambda b: b[0])
    for i, work in tqdm(dl, total=len(todo), desc="test whole-core segmentation"):
        r = todo.iloc[i]; iid = r.image_id
        if work is None or min(work.shape[:2]) < CFG.SEG_CROP:
            slog.add(iid, dict(_skip=True)); continue
        t_core = time.time(); mz = np.load(MAP_DIR / f"{iid}.npz")
        mp = ENGINE.maps(work, mz["I0"], mz["tissue"], mz["rim"])
        tis_t = mp["tissue"][0, 0]; tis = tis_t.cpu().numpy(); rim = mp["rim"][0, 0].cpu().numpy(); dab = mp["DAB"][0, 0].cpu().numpy()
        pm, pe, sd, boxes = segment_core(SEG_P, work, tis_t, CFG.TTA)
        mem = remove_small((pm >= 0.5) & tis, CFG.MIN_MEM_OBJ_PX)
        grade = np.zeros(mem.shape, np.uint8)
        for g, k in enumerate(("weak", "mod", "strong"), 1): grade[mem & (dab >= CALIB[f"t_{k}"])] = g
        rule = mp["mem_weak"][0, 0].cpu().numpy() & ~rim
        interior = tis & ~rim; nt = max(interior.sum(), 1)
        rec = dict(image_id=iid, group=r.group, ihc_score=r.ihc_score, her2_status=r.her2_status,
                   stu_mem_any=float((mem & interior).sum() / nt), stu_mem_weak=float(((grade >= 1) & interior).sum() / nt),
                   stu_mem_mod=float(((grade >= 2) & interior).sum() / nt), stu_mem_strong=float(((grade >= 3) & interior).sum() / nt),
                   rule_mem_weak=float((rule & interior).sum() / nt), stu_vs_rule_dice=dice_np(mem & interior, rule & interior),
                   rim_to_interior_mem_ratio=float(((grade >= 1) & rim).sum() / max(rim.sum(), 1) / max(((grade >= 1) & interior).sum() / nt, 1e-6)),
                   evidence_area=float(((pe >= 0.5) & tis).sum() / max(tis.sum(), 1)), mean_tta_sd=float(sd[tis].mean()) if tis.any() else np.nan,
                   subthreshold_membrane=float((mem & (grade == 0) & interior).sum() / nt))
        rec["seg_seconds"] = float(time.time() - t_core)          # maps + segmentation + features per core (decode excluded; TTA included; stain-shift rerun excluded)
        if iid in perturb_ids:
            xs = torch.from_numpy(work).to(DEVICE).permute(2, 0, 1)[None].float() / 255
            od = -torch.log(xs.clamp(min=1 / 255)); c = torch.einsum("bchw,cd->bdhw", od, M_STAIN_INV)
            c = c * torch.tensor([0.95, 1.05, 1.0], device=DEVICE).view(1, 3, 1, 1)
            wp = (torch.exp(-torch.einsum("bdhw,dc->bchw", c, M_STAIN)).clamp(0, 1)[0].permute(1, 2, 0) * 255).byte().cpu().numpy()
            pm2, _, _, _ = segment_core(SEG_P, wp, tis_t, 1)
            rec["stain_shift_dice"] = dice_np(mem, remove_small((pm2 >= 0.5) & tis, CFG.MIN_MEM_OBJ_PX))
        core_rois = []; S = CFG.SEG_CROP
        for (y, x) in boxes:
            ti = interior[y:y + S, x:x + S]
            if ti.mean() < 0.4: continue
            core_rois.append(dict(image_id=iid, y=int(y), x=int(x), tissue=float(ti.mean()), tta_sd=float(sd[y:y + S, x:x + S][ti].mean()),
                             evidence=float(pe[y:y + S, x:x + S][ti].mean()), mem=float(mem[y:y + S, x:x + S][ti].mean())))
        if CFG.SAVE_FULLRES_MASKS and free_disk_gb() < 3:
            CFG.SAVE_FULLRES_MASKS = False; print("!! free disk < 3 GB: full-resolution test masks are no longer saved (overlays still are)")
        if CFG.SAVE_FULLRES_MASKS:
            cv2.imwrite(str(TEST_OUT / f"{iid}_membrane_grade.png"), grade)
            cv2.imwrite(str(TEST_OUT / f"{iid}_p_membrane.png"), (pm * 255).astype(np.uint8))
            cv2.imwrite(str(TEST_OUT / f"{iid}_p_evidence.png"), (pe * 255).astype(np.uint8))
        R = 4; small = cv2.resize(work, (work.shape[1] // R, work.shape[0] // R), interpolation=cv2.INTER_AREA)
        gs = cv2.resize(grade, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = small.copy()
        for g, col in [(1, (42, 120, 214)), (2, (235, 104, 52)), (3, (227, 73, 72))]: ov[gs == g] = col
        cv2.imwrite(str(TEST_OUT / f"{iid}_overlay.jpg"), cv2.cvtColor(np.concatenate([small, ov], 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        del mp
        slog.add(iid, dict(rec=rec, rois=core_rois))              # written last: overlay/masks above are idempotent rewrites
    done = [x for x in slog.records() if not x.get("_skip")]
    TSEG = pd.DataFrame([x["rec"] for x in done])
    ROIC = pd.DataFrame([roi for x in done for roi in x["rois"]],
                        columns=["image_id", "y", "x", "tissue", "tta_sd", "evidence", "mem"])
    atomic_csv(TSEG, TEST_SEG_PATH, index=False); atomic_csv(ROIC, ROI_PATH, index=False)
section_time("whole-core test segmentation")

# ablation students on the same test proxies (crop-level, cheaper): IHC-0 membranous area >= moderate, TTA stability
@torch.no_grad()
def student_proxy_on_test(model, n_cores=60, tta=CFG.TTA):
    sub = TE.sample(min(n_cores, len(TE)), random_state=CFG.SEED).reset_index(drop=True); out = []
    S = CFG.SEG_CROP; rng = np.random.default_rng(1)
    for _, r in sub.iterrows():
        work = read_rgb(r.path, CFG.WORK_DS); mz = np.load(MAP_DIR / f"{r.image_id}.npz")
        mp = ENGINE.maps(work, mz["I0"], mz["tissue"], mz["rim"], need_ridge=False)
        c = BAGS[r.image_id]["coords"]; c = c[rng.choice(len(c), min(4, len(c)), replace=False)]
        H, W = work.shape[:2]; bx = [(int(np.clip(y, 0, H - S)), int(np.clip(x, 0, W - S))) for y, x in c]
        xb = torch.stack([torch.from_numpy(work[y:y + S, x:x + S]) for y, x in bx]).to(DEVICE).permute(0, 3, 1, 2).float() / 255
        pm, _, sd = seg_predict(model, xb, tta)
        d = torch.stack([mp["DAB"][0, 0, y:y + S, x:x + S] for y, x in bx]); t = torch.stack([mp["tissue"][0, 0, y:y + S, x:x + S] for y, x in bx])
        m = (pm >= 0.5) & t
        out.append(dict(group=r.group, mem_mod=float((m & (d >= CALIB["t_mod"])).sum() / t.sum().clamp(min=1)), tta_sd=float(sd[t].mean()) if t.any() else np.nan))
    o = pd.DataFrame(out)
    return dict(IHC0_membranous_area_ge_mod=float(o[o.group == "0"].mem_mod.mean()) if (o.group == "0").any() else np.nan,
                IHC3_membranous_area_ge_mod=float(o[o.group == "3"].mem_mod.mean()) if (o.group == "3").any() else np.nan,
                mean_TTA_SD=float(o.tta_sd.mean()))
SEG_PROXY_PATH = CACHE / "seg_test_proxies.csv"
SEG_PROXY = pd.read_csv(SEG_PROXY_PATH) if SEG_PROXY_PATH.exists() else None
if SEG_PROXY is None or set(SEG_PROXY.model) != set(SEG_MODELS):
    SEG_PROXY = pd.DataFrame([{"model": n, **student_proxy_on_test(m)} for n, m in SEG_MODELS.items()])
    SEG_PROXY["IHC3_to_IHC0_contrast"] = SEG_PROXY.IHC3_membranous_area_ge_mod / SEG_PROXY.IHC0_membranous_area_ge_mod.clip(lower=1e-6)
    atomic_csv(SEG_PROXY, SEG_PROXY_PATH, index=False)
SEG_PROXY.to_csv(TAB / "seg_test_proxies.csv", index=False); print(SEG_PROXY.round(4).to_string(index=False))
print("Test whole-core summary by IHC group:\n", TSEG.groupby("group")[["stu_mem_weak", "stu_mem_mod", "stu_mem_strong", "rule_mem_weak",
      "stu_vs_rule_dice", "rim_to_interior_mem_ratio", "mean_tta_sd"]].median().reindex(GROUP_ORDER).round(4))
if "stain_shift_dice" in TSEG: print("stain-shift self-consistency Dice (median):", round(float(TSEG.stain_shift_dice.median()), 3))

# %% [markdown]
# ## 16. Clinical consistency of the segmentation-derived features (the downstream proxy)
# If the membrane maps capture diagnostically relevant structure, the features derived from them should do three things:
# 1. increase monotonically with the IHC score (Spearman ρ with a bootstrap CI, plus Kruskal–Wallis)
# 2. predict FISH-resolved status on the **independent test set**, when an interpretable logistic model is fitted on the train cores
# 3. beat the stain-intensity rule
#
# Features from the rule map, the student map, and the student plus cell morphometry are compared. This is evidence of *clinical plausibility*,
# not of pixel accuracy.

# %%
FEATSETS = {
    "rule membrane fractions": [c for c in COREF.columns if c.startswith("rule_mem_")],
    "student membrane fractions": [c for c in COREF.columns if c.startswith("stu_mem_")],
    "student + cell morphometry": [c for c in COREF.columns if c.startswith("stu_")],
    "rule + cell morphometry": [c for c in COREF.columns if c.startswith("rule_")],
}
cf_tr = COREF[COREF.split == "train"].merge(TR[["image_id", "fold"]], on="image_id")
cf_te = COREF[COREF.split == "test"].set_index("image_id").reindex(TE.image_id).reset_index()
CLIN = []
for fs, cols in FEATSETS.items():
    Xtr = cf_tr[cols].fillna(0).values; Xte = cf_te[cols].fillna(0).values
    mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)).fit(Xtr, cf_tr.her2_status)
    p = mdl.predict_proba(Xte)[:, 1]; ok = ~np.isnan(cf_te[cols].values).any(1)
    auc = roc_auc_score(cf_te.her2_status[ok], p[ok]) if len(set(cf_te.her2_status[ok])) > 1 else np.nan
    lo, hi = boot_ci(cf_te.her2_status[ok].values, p[ok], 0.5)
    CLIN.append(dict(feature_set=fs, n_features=len(cols), test_AUROC=auc, lo=lo["AUROC"], hi=hi["AUROC"]))
CLIN.append(dict(feature_set="Stain intensity rule (Pisula 2023)", n_features=1, test_AUROC=CLS[CLS.method == "Stain intensity rule (Pisula 2023)"].AUROC.iloc[0],
                 lo=CLS[CLS.method == "Stain intensity rule (Pisula 2023)"].AUROC_lo.iloc[0], hi=CLS[CLS.method == "Stain intensity rule (Pisula 2023)"].AUROC_hi.iloc[0]))
CLIN = pd.DataFrame(CLIN); CLIN.to_csv(TAB / "clinical_feature_models.csv", index=False); print(CLIN.round(3).to_string(index=False))

TREND = []
key_feats = ["rule_mem_mod_frac", "stu_mem_mod_frac", "stu_mem_strong_frac", "stu_cell_cov>=0.5", "stu_cell_cov>=0.5&mod", "stu_cell_median_cov_pos"]
for split, d in [("train", cf_tr), ("test", cf_te)]:
    for f in key_feats:
        if f not in d: continue
        x, y = d[f].values, d.ihc_score.values; ok = ~np.isnan(x)
        rho = stats.spearmanr(y[ok], x[ok]).correlation; br = []
        r0 = np.random.default_rng(0)
        for _ in range(500):
            ii = r0.choice(ok.sum(), ok.sum()); br.append(stats.spearmanr(y[ok][ii], x[ok][ii]).correlation)
        groups = [x[ok][d.group.values[ok] == g] for g in GROUP_ORDER if (d.group.values[ok] == g).sum() > 1]
        kw = stats.kruskal(*groups).pvalue if len(groups) > 1 else np.nan
        med = {f"median_{g}": float(np.median(x[ok][d.group.values[ok] == g])) if (d.group.values[ok] == g).any() else np.nan for g in GROUP_ORDER}
        TREND.append(dict(split=split, feature=f, spearman_rho=rho, rho_lo=np.nanquantile(br, 0.025), rho_hi=np.nanquantile(br, 0.975), kruskal_p=kw, **med))
TREND = pd.DataFrame(TREND); TREND.to_csv(TAB / "clinical_feature_trends.csv", index=False); print(TREND.round(3).to_string(index=False))

# ---- H3: do membrane / cell features separate the clinically hard pairs? ----
# (a) IHC 3+ vs IHC 2+ (FISH- and FISH+ pooled): the visual 2+/3+ boundary that the ASCO/CAP score rests on.
# (b) 2+/FISH+ vs 2+/FISH-: whether morphometry carries any signal for the equivocal cores that currently need ISH.
# Single-feature AUROC (direction fixed on TRAIN, so the test AUROC is not optimistically flipped), stratified bootstrap CI,
# Holm correction across features within each comparison x split. Small groups are reported, never hidden.
def strat_boot_auc(y, x, n=2000, seed=0):
    """Stratified bootstrap CI of AUROC, vectorised via the Mann-Whitney rank identity (ties get mid-ranks)."""
    r0 = np.random.default_rng(seed); x1, x0 = x[y == 1], x[y == 0]; n1, n0 = len(x1), len(x0)
    X = np.concatenate([x1[r0.integers(0, n1, (n, n1))], x0[r0.integers(0, n0, (n, n0))]], 1)
    R = stats.rankdata(X, axis=1)
    auc = (R[:, :n1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.quantile(auc, [0.025, 0.975])
def mw_p(y, x):
    return stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided").pvalue
H3_FEATS = [c for c in COREF.columns if c.startswith(("stu_", "rule_")) and COREF[c].dtype.kind in "fi"]
H3 = []
for comp, pos, neg in [("3+ vs 2+", ["3"], ["2-", "2+"]), ("2+/FISH+ vs 2+/FISH-", ["2+"], ["2-"])]:
    dtr = cf_tr[cf_tr.group.isin(pos + neg)]; dte = cf_te[cf_te.group.isin(pos + neg)]
    for f in H3_FEATS:
        xtr = dtr[f].values.astype(float); ytr = dtr.group.isin(pos).values.astype(int); ok = ~np.isnan(xtr)
        if ok.sum() < 6 or len(set(ytr[ok])) < 2 or np.nanstd(xtr) == 0: continue
        sign = 1.0 if roc_auc_score(ytr[ok], xtr[ok]) >= 0.5 else -1.0          # direction learned on train only
        for split, d in [("train", dtr), ("test", dte)]:
            x = sign * d[f].values.astype(float); y = d.group.isin(pos).values.astype(int); ok = ~np.isnan(x)
            if ok.sum() < 4 or len(set(y[ok])) < 2: continue
            lo, hi = strat_boot_auc(y[ok], x[ok], n=1000 if split == "train" else 2000)
            H3.append(dict(comparison=comp, split=split, feature=f, direction="+" if sign > 0 else "-", n_pos=int(y[ok].sum()),
                           n_neg=int((1 - y[ok]).sum()), AUROC=roc_auc_score(y[ok], x[ok]), lo=lo, hi=hi, p_mannwhitney=mw_p(y[ok], x[ok])))
H3 = pd.DataFrame(H3)
if len(H3):
    H3["p_holm"] = np.nan
    for _, idx in H3.groupby(["comparison", "split"]).groups.items():
        p = H3.loc[idx, "p_mannwhitney"].values; o = np.argsort(p); m = len(p); adj = np.empty(m)
        adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); H3.loc[idx, "p_holm"] = adj
    H3 = H3.sort_values(["comparison", "split", "AUROC"], ascending=[True, False, False])
    H3.to_csv(TAB / "h3_feature_separation_3v2_and_fish.csv", index=False)
    for (comp, split), d in H3.groupby(["comparison", "split"], sort=False):
        print(f"\nH3 {comp} [{split}] (n+={d.n_pos.iloc[0]}, n-={d.n_neg.iloc[0]}) top features:")
        print(d.head(6)[["feature", "direction", "AUROC", "lo", "hi", "p_holm"]].round(3).to_string(index=False))
    # figure: train vs test AUROC of every feature for the 3+ vs 2+ comparison (features ranked on train)
    d = H3[H3.comparison == "3+ vs 2+"].pivot_table(index="feature", columns="split", values=["AUROC", "lo", "hi"])
    if ("AUROC", "test") in d.columns:
        d = d.sort_values(("AUROC", "train"), ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2)); yy = np.arange(len(d))
        for off, sp, col in [(-0.15, "train", "#2a78d6"), (0.15, "test", "#e34948")]:
            ax.errorbar(d[("AUROC", sp)], yy + off, xerr=[d[("AUROC", sp)] - d[("lo", sp)], d[("hi", sp)] - d[("AUROC", sp)]],
                        fmt="o", ms=4, color=col, label=sp, capsize=2)
        ax.axvline(0.5, color="#888", ls="--", lw=1); ax.set_yticks(yy); ax.set_yticklabels(d.index, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("AUROC, IHC 3+ vs 2+ (95% stratified bootstrap CI)"); ax.legend(fontsize=8)
        savefig(fig, "fig12_h3_3plus_vs_2plus")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, f in zip(axes, ["stu_mem_mod_frac", "stu_cell_cov>=0.5", "stu_cell_cov>=0.5&mod", "rule_mem_mod_frac"]):
    for k, g in enumerate(GROUP_ORDER):
        v = cf_te[cf_te.group == g][f].dropna().values if f in cf_te else []
        if len(v) == 0: v = cf_tr[cf_tr.group == g][f].dropna().values; mk = "x"
        else: mk = "o"
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v, s=8, color=GROUP_COL[g], alpha=0.6, marker=mk)
        if len(v): ax.hlines(np.median(v), k - 0.3, k + 0.3, color="#0b0b0b", lw=1.5)
    ax.set_xticks(range(len(GROUP_ORDER))); ax.set_xticklabels(GROUP_ORDER); ax.set_xlabel("IHC group"); ax.set_title(f, fontsize=8)
axes[0].set_ylabel("fraction per core")
savefig(fig, "fig10_clinical_consistency")
section_time("clinical consistency")
