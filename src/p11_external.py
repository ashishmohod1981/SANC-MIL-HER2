# %% [markdown]
# ## 21. External validation and ablation on an independent dataset (optional)
# **Purpose.** This section tests whether the models and measurements trained on the gastro-oesophageal TMA cores transfer to
# data from another centre, scanner and organ. Nothing is re-trained: the stain matrix, DAB thresholds, tile encoder, MIL fold
# models and segmentation student are applied **frozen**.
#
# **Supported datasets** (the file layout is detected automatically; a generic CSV also works):
# | Dataset | Content | What it tests |
# |---|---|---|
# | Warwick HER2 Scoring Contest (breast, HER2 IHC whole slides, NanoZoomer) | HER2 score 0–3+ and the pathologists' **percentage of cells with complete membrane staining** (PCMS) | Membrane-coverage features against an expert membrane measure; status; H1 on IHC 0 slides |
# | ADEL (breast, HER2 IHC images or tiles, CC BY 4.0) | HER2 expression level per image | Status and monotonicity with score; H1 on IHC 0 images |
# | Generic | a CSV with `path`, `ihc_score` and optionally `her2_status`, `pcms`, `mpp` | Same, as available |
#
# **Analyses** (all pre-specified here, before seeing external results):
# 1. **Status**: AUROC of SANC-MIL, ABMIL, the stain-intensity rule and the membrane-feature model. Positive = IHC 3+ or 2+/ISH+;
#    negative = IHC 0/1+. IHC 2+ without an ISH result is excluded from status metrics.
# 2. **Monotonicity**: Spearman ρ between IHC score and P(HER2+), the expected ordinal score, and membrane features.
# 3. **H1 replication (ablation)**: tile false-positive rate in external IHC 0 images, SANC-MIL vs the same model trained without the
#    negative-bag constraint; paired Wilcoxon and bootstrap CI.
# 4. **Membrane measure vs experts** (Warwick only): Spearman ρ between PCMS and the fraction of cells with ≥70% membrane coverage
#    (primary, the closest analogue of "complete"); other coverage bins as secondary.
# 5. **Stain-calibration ablation**: every analysis is run twice, with the training stain matrix (frozen) and with a stain matrix
#    re-estimated on the external images (no labels used). DAB thresholds stay frozen in both.
#
# **Caveat to state in the paper.** Breast HER2 scoring requires complete, circumferential membrane staining; gastric scoring also
# counts basolateral and lateral staining. The external results are therefore a cross-organ transfer test, not a validation of
# gastric scoring rules.
#
# **How to use.** Attach the dataset with *Add Input*, set `EXT_ROOT` (and `EXT_MPP` for plain images without scale metadata),
# then run this cell. Results go to `tables/external_<name>_*.csv`, `figures/fig13_external_<name>.png` and
# `external_<name>_results.zip`. The cell resumes per image if interrupted.

# %%
try: import openslide                                     # whole-slide reader (Warwick .ndpi); not needed for plain images
except ImportError:
    try: _pip("openslide-bin"); _pip("openslide-python")
    except Exception as e: print("openslide unavailable (whole slides cannot be read):", e)
EXT_ROOT = os.environ.get("HER2_EXT_ROOT")          # e.g. "/kaggle/input/warwick-her2" or "/kaggle/input/adel-her2"
EXT_NAME = os.environ.get("HER2_EXT_NAME")          # "warwick" | "adel" | "generic" (auto-detected when None)
EXT_CSV = os.environ.get("HER2_EXT_CSV")            # generic manifest (path, ihc_score[, her2_status, pcms, mpp])
EXT_MPP = float(os.environ["HER2_EXT_MPP"]) if os.environ.get("HER2_EXT_MPP") else None   # µm/px of plain images, if known
EXT_STAIN_MODES = ("frozen", "refit")
EXT_MAX_REGIONS = 16 if RUN_MODE != "debug" else 4  # regions of 6x6 tiles sampled per slide (whole slides only)
EXT_FEAT_CROPS = CFG.FEAT_CROPS_PER_CORE
EXT_WSI_MIN_PX = int(os.environ.get("HER2_EXT_WSI_MIN_PX", 12000))   # a .tif larger than this (per side) is read as a whole slide

def _find_cols(df, *keys):
    """First column whose normalised name contains all keys (case/space/punctuation-insensitive)."""
    norm = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns}
    for c, n in norm.items():
        if all(k in n for k in keys): return c
    return None

def _score_to_int(v):
    s = str(v).strip().lower().replace("+", "")
    m = re.search(r"[0-3]", s)
    return int(m.group()) if m else np.nan

WSI_EXT = {".ndpi", ".svs", ".mrxs", ".scn", ".vms", ".bif", ".tiff", ".tif", ".kfb"}
IMG_EXT2 = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _read_table(p):
    p = Path(p)
    return pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)

def build_manifest(root, name=None, csv=None):
    """Return a DataFrame with path, ihc_score, her2_status (NaN if unknown), pcms (NaN if absent), mpp (NaN if unknown)."""
    root = Path(root)
    if csv:
        m = _read_table(csv)
        pc = _find_cols(m, "path") or _find_cols(m, "file") or _find_cols(m, "image")
        out = pd.DataFrame({"path": [str(root / str(x)) if not os.path.isabs(str(x)) else str(x) for x in m[pc]],
                            "ihc_score": m[_find_cols(m, "score") or _find_cols(m, "her2")].map(_score_to_int)})
        for col, keys in [("her2_status", ("status",)), ("pcms", ("percent",)), ("mpp", ("mpp",))]:
            c = _find_cols(m, *keys); out[col] = pd.to_numeric(m[c], errors="coerce") if c else np.nan
        return out.assign(dataset=name or "generic")
    tables = [p for p in root.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (WSI_EXT | IMG_EXT2)]
    # --- Warwick: whole slides named like "<case>_HER2.ndpi" (H&E slides are ignored) + a ground-truth table with score + percentage ---
    for t in tables:
        g = _read_table(t)
        sc, pc = _find_cols(g, "score"), _find_cols(g, "percent")
        if sc and pc:
            case = _find_cols(g, "case") or g.columns[0]
            her2 = [p for p in files if p.suffix.lower() in WSI_EXT and re.search(r"her2", p.name, re.I)]
            rows = []
            for _, r in g.iterrows():
                cid = re.sub(r"\D", "", str(r[case]))
                key = lambda p: re.sub(r"\D", "", re.split(r"her2", p.stem, flags=re.I)[0]) or re.sub(r"\D", "", p.parent.name)
                hit = [p for p in her2 if key(p).lstrip("0") == cid.lstrip("0") and cid]
                if hit: rows.append(dict(path=str(hit[0]), ihc_score=_score_to_int(r[sc]), her2_status=np.nan,
                                         pcms=pd.to_numeric(r[pc], errors="coerce"), mpp=np.nan))
            if rows:
                print(f"external: Warwick-style layout ({t.name}): {len(rows)} HER2 slides with score + percentage")
                return pd.DataFrame(rows).assign(dataset=name or "warwick")
    # --- ADEL / generic image folders: a labels table naming image files (or folders of tiles) + a HER2 level ---
    for t in tables:
        g = _read_table(t)
        lc = _find_cols(g, "her2") or _find_cols(g, "label") or _find_cols(g, "score") or _find_cols(g, "class")
        fc = _find_cols(g, "file") or _find_cols(g, "image") or _find_cols(g, "name") or _find_cols(g, "id")
        if not (lc and fc): continue
        by_stem = defaultdict(list)
        for p in files: by_stem[p.stem].append(p)
        dirs = {p.name: p for p in root.rglob("*") if p.is_dir()}
        rows = []
        for _, r in g.iterrows():
            key = Path(str(r[fc])).stem
            if key in by_stem: path = str(by_stem[key][0])
            elif key in dirs: path = str(dirs[key])                         # a folder of tiles from one slide
            else: continue
            rows.append(dict(path=path, ihc_score=_score_to_int(r[lc]), her2_status=np.nan, pcms=np.nan, mpp=np.nan))
        if rows:
            print(f"external: labelled image layout ({t.name}): {len(rows)} images or tile folders")
            return pd.DataFrame(rows).assign(dataset=name or "adel")
    raise RuntimeError(f"No usable label table found under {root}. Provide HER2_EXT_CSV with columns path, ihc_score[, her2_status, pcms, mpp].")

# ---------------- reading: whole slide, single image, or folder of tiles -> work-level regions ----------------
def _open_slide(path):
    import openslide
    return openslide.OpenSlide(str(path))

def _is_wsi(path):
    p = Path(path)
    if p.is_dir() or p.suffix.lower() not in WSI_EXT: return False
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(p) as im: return im.size[0] * im.size[1] > EXT_WSI_MIN_PX ** 2
        except Exception: return True
    return True

def estimate_mpp_from_nuclei(rgb, M):
    """Fallback scale estimate when an image carries no µm/px: median equivalent diameter of haematoxylin-dominant nuclei,
    assuming 7 µm for a typical carcinoma nucleus. Reported, never silently trusted."""
    od = od_np(rgb, estimate_I0(rgb)).reshape(-1, 3)
    c = np.clip(od @ np.linalg.pinv(np.asarray(M, np.float32)), 0, None).reshape(rgb.shape[0], rgb.shape[1], 3)
    hmask = (c[..., 0] > np.percentile(c[..., 0], 85)) & (c[..., 0] > c[..., 1])
    hmask = ndi.binary_opening(hmask, iterations=1)
    lab, n = ndi.label(hmask)
    if n < 20: return np.nan
    areas = np.bincount(lab.ravel())[1:]; areas = areas[(areas > 10) & (areas < np.percentile(areas, 95))]
    return 7.0 / (2 * np.sqrt(np.median(areas) / np.pi)) if len(areas) else np.nan

def iter_regions(row, rng, region_px=6 * CFG.TILE):
    """Yield (work-level RGB region, I0) for one external item; at most EXT_MAX_REGIONS regions per slide."""
    path = Path(row.path)
    if path.is_dir():                                           # folder of tiles: each tile is a region
        tiles = sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXT2])
        if len(tiles) > EXT_MAX_REGIONS * 9: tiles = [tiles[i] for i in np.sort(rng.choice(len(tiles), EXT_MAX_REGIONS * 9, replace=False))]
        imgs = [np.asarray(Image.open(t).convert("RGB")) for t in tiles]
        mpp = row.mpp if np.isfinite(row.mpp) else EXT_MPP
        I0 = estimate_I0(np.concatenate([i.reshape(-1, 3) for i in imgs])[None])
        for im in imgs:
            f = mpp / WORK_MPP
            yield cv2.resize(im, (max(1, int(im.shape[1] * f)), max(1, int(im.shape[0] * f))), interpolation=cv2.INTER_AREA), I0
        return
    if _is_wsi(path):
        sl = _open_slide(path)
        mpp = float(sl.properties.get("openslide.mpp-x", "nan")) if np.isnan(row.mpp) else row.mpp
        if not np.isfinite(mpp): mpp = EXT_MPP or 0.25
        W0, H0 = sl.dimensions
        th = np.asarray(sl.get_thumbnail((max(1, W0 // 32), max(1, H0 // 32))).convert("RGB")); sx = W0 / th.shape[1]
        I0 = estimate_I0(th)
        om = od_np(th, I0).max(-1); thr = float(np.clip(np.percentile(om, 10) + 0.045, *CFG.OD_TISSUE_RANGE))
        tis = cv2.morphologyEx((om > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        side0 = region_px * WORK_MPP / mpp                       # region side in level-0 pixels
        step = max(1, int(side0 / sx))
        cand = [(y, x) for y in range(0, tis.shape[0] - step, step) for x in range(0, tis.shape[1] - step, step)
                if tis[y:y + step, x:x + step].mean() > 0.6]
        if not cand: return
        cand = [cand[i] for i in np.sort(rng.choice(len(cand), min(EXT_MAX_REGIONS, len(cand)), replace=False))]
        level = sl.get_best_level_for_downsample(max(1.0, WORK_MPP / mpp))
        ds = sl.level_downsamples[level]
        for y, x in cand:
            X0, Y0 = int(x * sx), int(y * sx); sz = int(round(side0 / ds))
            reg = np.asarray(sl.read_region((X0, Y0), level, (sz, sz)).convert("RGB"))
            yield cv2.resize(reg, (region_px, region_px), interpolation=cv2.INTER_AREA), I0
        sl.close(); return
    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(path).convert("RGB"))             # single image (tile, core or large field)
    mpp = row.mpp if np.isfinite(row.mpp) else EXT_MPP
    if mpp is None or not np.isfinite(mpp):
        raise RuntimeError("Plain images need a scale: set HER2_EXT_MPP (µm/px) or an mpp column (see the scale check printed above).")
    f = mpp / WORK_MPP; I0 = estimate_I0(im)
    work = cv2.resize(im, (max(1, int(im.shape[1] * f)), max(1, int(im.shape[0] * f))), interpolation=cv2.INTER_AREA); del im
    H, W = work.shape[:2]
    if H <= 2 * region_px and W <= 2 * region_px:
        yield work, I0; return
    om = od_np(thumb_of(work), I0).max(-1) > 0.1                # large image: sample tissue regions like a whole slide
    R = CFG.THUMB_REL; step = region_px // R
    cand = [(y, x) for y in range(0, om.shape[0] - step, step) for x in range(0, om.shape[1] - step, step) if om[y:y + step, x:x + step].mean() > 0.6]
    for i in np.sort(rng.choice(len(cand), min(EXT_MAX_REGIONS, len(cand)), replace=False)) if cand else []:
        y, x = cand[i]; yield work[y * R:y * R + region_px, x * R:x * R + region_px], I0

# ---------------- per-region features, identical in definition to pass 1 (Section 4) ----------------
@torch.no_grad()
def region_tile_features(engine, work, I0):
    T = CFG.TILE; H, W = work.shape[:2]
    if H < T or W < T: return None
    mp = engine.maps(work, I0, None, None)
    ny, nx = H // T, W // T
    cut = lambda m: m.float()[..., :ny * T, :nx * T]
    pool = lambda m: F.avg_pool2d(cut(m), T, T)[0, 0]
    tis = mp["tissue"]; tf = pool(tis); tisN = tf.clamp(min=1e-6)
    frac = lambda m: pool(m & tis) / tisN
    dab = mp["DAB"] * tis; dmean = pool(dab) / tisN; dsq = pool(dab * dab) / tisN
    lap = F.conv2d(F.pad(mp["gray"], (1, 1, 1, 1), mode="reflect"), LAP); focus = pool(lap * lap) - pool(lap) ** 2
    feats = torch.stack([tf, dmean, torch.sqrt((dsq - dmean ** 2).clamp(min=0)),
                         frac(mp["DAB"] >= CALIB["t_weak"]), frac(mp["DAB"] >= CALIB["t_mod"]), frac(mp["DAB"] >= CALIB["t_strong"]),
                         frac(mp["mem_weak"]), frac(mp["mem_mod"]), frac(mp["mem_strong"]), frac(mp["nonmem"]),
                         frac(mp["nuc"] & (mp["DAB"] >= CALIB["t_weak"])), frac(mp["nuc"]), focus, torch.zeros_like(tf)], -1)
    assert feats.shape[-1] == len(HAND_NAMES), "feature definition drifted from pass 1"
    keep = torch.nonzero(tf >= CFG.MIN_TILE_TISSUE).cpu().numpy()
    return mp, feats[keep[:, 0], keep[:, 1]].cpu().numpy().astype(np.float32) if len(keep) else np.zeros((0, len(HAND_NAMES)), np.float32), keep * T

@torch.no_grad()
def process_item(row, engine, enc, rng):
    """Tiles -> embeddings + stain features (for MIL) and 512-px crops -> membrane / cell features (as in Section 14)."""
    embs, hands, S = [], [], CFG.SEG_CROP; acc = defaultdict(float); cov = {"stu": [[], []], "rule": [[], []]}; n_crops = 0
    for work, I0 in iter_regions(row, rng):
        out = region_tile_features(engine, work, I0)
        if out is None: continue
        mp, hand, coords = out
        if len(coords):
            xg = torch.from_numpy(np.ascontiguousarray(work)).to(DEVICE).permute(2, 0, 1); T = CFG.TILE
            for b in range(0, len(coords), CFG.ENC_BATCH):
                cb = coords[b:b + CFG.ENC_BATCH]
                batch = torch.stack([xg[:, y:y + T, x:x + T] for y, x in cb]).float() / 255
                with autocast(): embs.append(gpu_safe(enc, batch).float().cpu().numpy().astype(np.float16))
            hands.append(hand)
        H, W = work.shape[:2]
        if n_crops < EXT_FEAT_CROPS and H >= S and W >= S:      # one 512-px crop per region for cell morphometry
            tis = mp["tissue"][0, 0]
            y, x = (H - S) // 2, (W - S) // 2
            ti = tis[y:y + S, x:x + S].cpu().numpy()
            if ti.mean() > 0.3:
                xb = torch.from_numpy(np.ascontiguousarray(work[y:y + S, x:x + S]))[None].to(DEVICE).permute(0, 3, 1, 2).float() / 255
                pmem, _, _ = seg_predict(SEG_P, xb, tta=1)
                d = mp["DAB"][0, 0, y:y + S, x:x + S].cpu().numpy(); hh = mp["H"][0, 0, y:y + S, x:x + S].cpu().numpy()
                stu = remove_small((pmem[0].cpu().numpy() >= 0.5) & ti, CFG.MIN_MEM_OBJ_PX)
                rule = mp["mem_weak"][0, 0, y:y + S, x:x + S].cpu().numpy() & ti
                acc["tissue_px"] += ti.sum()
                for nm, mk in [("stu", stu), ("rule", rule)]:
                    for k in ("weak", "mod", "strong"): acc[f"{nm}_mem_{k}"] += (mk & (d >= CALIB[f"t_{k}"])).sum()
                res = cell_analysis((hh, d, ti, [stu, rule]))
                for mi, nm in enumerate(["stu", "rule"]): cov[nm][0].append(res[mi][0]); cov[nm][1].append(res[mi][1])
                n_crops += 1
        del mp
    if not hands: return None
    hand = np.concatenate(hands); emb = np.concatenate(embs)
    if len(hand) > CFG.MAX_TILES_PER_IMAGE:
        k = np.sort(rng.choice(len(hand), CFG.MAX_TILES_PER_IMAGE, replace=False)); hand, emb = hand[k], emb[k]
    rec = {k + "_frac": v / max(acc["tissue_px"], 1) for k, v in acc.items() if k != "tissue_px"}
    for nm in ("stu", "rule"):
        c = np.concatenate(cov[nm][0]) if cov[nm][0] else np.zeros(0); dd = np.concatenate(cov[nm][1]) if cov[nm][1] else np.zeros(0)
        rec.update(summarize_cells(c, dd, f"{nm}_cell_"))
    if n_crops == 0:                                            # images too small for a 512-px crop: no membrane/cell features
        rec = {k: np.nan for k in rec}
    return dict(emb=emb, hand=hand, feats=rec, n_tiles=len(hand), n_crops=n_crops)

def _bag_tensors(item):
    h = ((transform_hand(item["hand"]) - HAND_MU) / HAND_SD).astype(np.float32)
    e = torch.from_numpy(item["emb"].astype(np.float32))[None].to(DEVICE); hd = torch.from_numpy(h)[None].to(DEVICE)
    return e, hd, torch.ones(1, e.shape[1], dtype=torch.bool, device=DEVICE)

@torch.no_grad()
def predict_external(models, item, T=1.0):
    e, hd, mk = _bag_tensors(item); outs = [m.eval()(e, hd, mk) for m in models]
    logit = float(np.mean([o["logit"].item() for o in outs])) / T
    inst = np.mean([o["inst"][0].cpu().numpy() for o in outs], 0) if outs[0]["inst"] is not None else None
    ordl = np.mean([o["ord"][0].cpu().numpy() for o in outs], 0) if "ord" in outs[0] else None
    return logit, inst, ordl

import hashlib
def run_external():
    if not EXT_ROOT or not Path(EXT_ROOT).exists():
        print("External validation skipped: set HER2_EXT_ROOT to an attached dataset (see the markdown above)."); return None
    man = build_manifest(EXT_ROOT, EXT_NAME, EXT_CSV)
    name = man.dataset.iloc[0]; man = man[man.ihc_score.notna()].reset_index(drop=True)
    man["item_id"] = [re.sub(r"[^A-Za-z0-9]+", "_", Path(p).stem)[:80] + "_" + hashlib.md5(str(p).encode()).hexdigest()[:6] for p in man.path]   # stable if rows are added
    print(f"external dataset '{name}': {len(man)} labelled items | score counts {man.ihc_score.value_counts().sort_index().to_dict()}")
    # scale check for plain images: estimated µm/px from nuclear size, printed so a wrong EXT_MPP is caught early
    first = Path(man.path.iloc[0])
    if not _is_wsi(first) and EXT_MPP is None and man.mpp.isna().all():   # only when no scale is given (the estimate is crude)
        f0 = first if first.is_file() else next(p for p in first.rglob("*") if p.suffix.lower() in IMG_EXT2)
        est = estimate_mpp_from_nuclei(np.asarray(Image.open(f0).convert("RGB")), CALIB["M"])
        print(f"scale check: nuclear-size estimate {est:.3f} µm/px (assumes 7 µm nuclei; UNRELIABLE on pale HER2 IHC nuclei - give the true scale)")
        if EXT_MPP is None and man.mpp.isna().all():
            print("!! No µm/px given; using the nuclear-size estimate. Set HER2_EXT_MPP for a reproducible result.")
            man["mpp"] = est
    use_embedding("emb")
    enc = TileEncoder(CFG.ENCODER).to(DEVICE).eval()
    variants = {"SANC-MIL": PROPOSED, "SANC without negative constraint": "SANC − negative constraint", "ABMIL": MAIN_BASELINE}
    MODELS = {k: fold_models(v) for k, v in variants.items()}   # unsaved variants are re-trained deterministically (fold seed = fold index)
    TEMPS = {k: float(CLS[CLS.method == v].temp.iloc[0]) for k, v in variants.items()}
    # interpretable comparators fitted on ALL training cores (as in Sections 8 and 16), applied unchanged
    rule_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(XS_tr[:, :1], ystat_tr)
    feat_cols = FEATSETS["student + cell morphometry"]
    feat_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)).fit(cf_tr[feat_cols].fillna(0).values, cf_tr.her2_status)
    engines = {"frozen": ENGINE}
    if "refit" in EXT_STAIN_MODES:                                 # unsupervised stain re-estimation on external thumbnails
        thumbs = []
        for _, r in man.sample(min(40, len(man)), random_state=CFG.SEED).iterrows():
            try:
                reg = next(iter_regions(r, np.random.default_rng(0)))
                thumbs.append((thumb_of(reg[0]), reg[1]))
            except Exception as e: print("refit: skipped", r.path, type(e).__name__)
        M_ext, minfo = macenko_global(thumbs)
        engines["refit"] = StainEngine(M_ext, CALIB); print("refit stain matrix vs Ruifrok:", minfo)
    out_rows = []
    for mode, eng in engines.items():
        log = ProgressLog(f"external_{name}_{mode}")
        for _, r in tqdm(man.iterrows(), total=len(man), desc=f"external {name} [{mode}]"):
            if r.item_id in log: continue
            rng = np.random.default_rng(_img_seed(r.item_id))
            try: item = process_item(r, eng, enc, rng)
            except Exception as e: print(f"  {r.item_id}: {type(e).__name__}: {str(e)[:120]}"); item = None
            if item is None: log.add(r.item_id, dict(_skip=True)); continue
            rec = dict(item_id=r.item_id, mode=mode, ihc_score=int(r.ihc_score), her2_status=r.her2_status, pcms=r.pcms, n_tiles=item["n_tiles"],
                       n_crops=item["n_crops"], **item["feats"])
            for k, ms in MODELS.items():
                lg, inst, ordl = predict_external(ms, item, TEMPS[k])
                rec[f"logit::{k}"] = lg
                if inst is not None: rec[f"tileFP::{k}"] = float(np.mean(sig(inst) > 0.5)); rec[f"tileMax::{k}"] = float(sig(inst).max())
                if ordl is not None: rec[f"exp_score::{k}"] = float((coral_probs(ordl[None]) * np.arange(4)).sum())
            h = item["hand"]; g = lambda n: h[:, HAND_NAMES.index(n)]
            rec["logit::Stain-intensity rule"] = float(rule_lr.decision_function([[g("dab_mean").max()]])[0])
            fv = [item["feats"].get(c, np.nan) for c in feat_cols]
            rec["logit::Membrane + cell features"] = float(feat_lr.decision_function([fv])[0]) if np.all(np.isfinite(fv)) else np.nan
            log.add(r.item_id, rec)
        out_rows += [x for x in log.records() if not x.get("_skip")]
    EXT = pd.DataFrame(out_rows).drop(columns=["_key"], errors="ignore")
    atomic_csv(EXT, TAB / f"external_{name}_per_item.csv", index=False)
    return name, EXT

if "pretty_feature" not in globals():          # defined in section 18c; fallback so this section also runs when 18c was skipped
    def pretty_feature(f):
        src = "student" if f.startswith("stu_") else ("rule" if f.startswith("rule_") else "")
        g = re.sub(r"^(stu|rule)_", "", f)
        names = {"mem_weak_frac": "membranous DAB area, ≥weak", "mem_mod_frac": "membranous DAB area, ≥moderate",
                 "mem_strong_frac": "membranous DAB area, ≥strong", "cell_cov>=0.3": "cells ≥30% membrane-covered",
                 "cell_cov>=0.5": "cells ≥50% membrane-covered", "cell_cov>=0.7": "cells ≥70% membrane-covered",
                 "cell_cov>=0.5&mod": "cells ≥50% covered at ≥moderate DAB", "cell_cov>=0.5&strong": "cells ≥50% covered at ≥strong DAB",
                 "cell_median_cov_pos": "median coverage of stained cells", "cell_n_cells": "number of cells"}
        return f"{names.get(g, g)} ({src})" if src else names.get(g, g)

def analyse_external(name, EXT):
    rows, h1_rows, pc_rows = [], [], []
    for mode, d in EXT.groupby("mode"):
        st = d.her2_status.where(d.her2_status.notna(), np.where(d.ihc_score == 3, 1.0, np.where(d.ihc_score <= 1, 0.0, np.nan)))
        ok = st.notna().values; y = st.values[ok].astype(int)
        for c in [c for c in d.columns if c.startswith("logit::")]:
            m = c.split("::")[1]; p = d[c].values.astype(float); fin = ok & np.isfinite(p)
            yy = st.values[fin].astype(int)
            r = dict(mode=mode, method=m, n_pos=int(yy.sum()), n_neg=int((1 - yy).sum()))
            if len(set(yy)) > 1:
                r["status_AUROC"] = roc_auc_score(yy, p[fin]); lo, hi = strat_boot_auc(yy, p[fin]); r["AUROC_lo"], r["AUROC_hi"] = lo, hi
            r["spearman_vs_score"] = stats.spearmanr(d.ihc_score, p, nan_policy="omit").correlation
            rows.append(r)
        for c in [c for c in d.columns if c.startswith("exp_score::")]:   # ordinal head: monotonicity AND status discrimination
            r = dict(mode=mode, method=c.split("::")[1] + " (expected ordinal score)", spearman_vs_score=stats.spearmanr(d.ihc_score, d[c]).correlation,
                     QWK_vs_score=cohen_kappa_score(d.ihc_score, np.clip(np.rint(d[c]), 0, 3).astype(int), weights="quadratic"))
            p = d[c].values.astype(float); fin = ok & np.isfinite(p); yy = st.values[fin].astype(int)
            if len(set(yy)) > 1:
                r.update(n_pos=int(yy.sum()), n_neg=int((1 - yy).sum()), status_AUROC=roc_auc_score(yy, p[fin]))
                r["AUROC_lo"], r["AUROC_hi"] = strat_boot_auc(yy, p[fin])
            rows.append(r)
        for f in ["stu_mem_mod_frac", "stu_mem_strong_frac", "stu_cell_cov>=0.5", "stu_cell_cov>=0.7", "stu_cell_cov>=0.5&mod"]:
            if f in d and d[f].notna().sum() > 3:
                rows.append(dict(mode=mode, method=f"feature: {pretty_feature(f)}", spearman_vs_score=stats.spearmanr(d.ihc_score, d[f], nan_policy="omit").correlation))
        z = d[d.ihc_score == 0]                                  # H1 replication on external IHC 0 items
        a, b = "tileFP::SANC-MIL", "tileFP::SANC without negative constraint"
        if len(z) >= 5 and a in z and b in z:
            h1_rows.append(dict(mode=mode, **paired_test(z[a].values, z[b].values),
                                any_tile_gt09_SANC=float((z["tileMax::SANC-MIL"] > 0.9).mean()),
                                any_tile_gt09_noconstraint=float((z["tileMax::SANC without negative constraint"] > 0.9).mean())))
        if d.pcms.notna().sum() >= 10:                            # Warwick: membrane coverage vs expert percentage
            q = d[d.pcms.notna() & d.get("stu_cell_cov>=0.7", pd.Series(np.nan, index=d.index)).notna()]
            for f in ["stu_cell_cov>=0.7", "stu_cell_cov>=0.5", "stu_cell_cov>=0.3", "rule_cell_cov>=0.7", "stu_cell_cov>=0.5&mod"]:
                if f not in q: continue
                rho = stats.spearmanr(q.pcms, q[f]).correlation; rb = np.random.default_rng(0)
                bs = [stats.spearmanr(*q.iloc[i][["pcms", f]].values.T).correlation for i in (rb.integers(0, len(q), len(q)) for _ in range(1000))]
                pc_rows.append(dict(mode=mode, feature=pretty_feature(f), primary=f == "stu_cell_cov>=0.7", n=len(q), spearman_rho=rho,
                                    lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975)))
    R, H1E, PC = pd.DataFrame(rows), pd.DataFrame(h1_rows), pd.DataFrame(pc_rows)
    atomic_csv(R, TAB / f"external_{name}_summary.csv", index=False)
    if len(H1E): atomic_csv(H1E, TAB / f"external_{name}_h1_negative_constraint.csv", index=False)
    if len(PC): atomic_csv(PC, TAB / f"external_{name}_pcms_agreement.csv", index=False)
    print(R.round(3).to_string(index=False))
    if len(H1E): print("\nH1 on external IHC 0 items (A = SANC-MIL, B = without negative constraint):\n", H1E.round(4).to_string(index=False))
    if len(PC): print("\nMembrane coverage vs pathologists' % of completely stained cells:\n", PC.round(3).to_string(index=False))
    # figure: score-wise distributions of P(HER2+) and the primary membrane feature, frozen stain mode
    d = EXT[EXT["mode"] == "frozen"]
    fig, ax = plt.subplots(1, 3 if len(PC) else 2, figsize=(12 if len(PC) else 8.5, 3.4))
    for j, (col, lab) in enumerate([("logit::SANC-MIL", "SANC-MIL P(HER2+)"), ("stu_cell_cov>=0.7", "cells ≥70% membrane-covered")]):
        for s in range(4):
            v = d[d.ihc_score == s][col].dropna().values
            if col.startswith("logit"): v = sig(v)
            ax[j].scatter(np.full(len(v), s) + np.random.default_rng(s).uniform(-0.18, 0.18, len(v)), v, s=10, alpha=0.6, color=PAL[s])
            if len(v): ax[j].hlines(np.median(v), s - 0.3, s + 0.3, color="#0b0b0b", lw=1.5)
        ax[j].set_xticks(range(4)); ax[j].set_xticklabels(["0", "1+", "2+", "3+"]); ax[j].set_xlabel("external HER2 score"); ax[j].set_title(lab, fontsize=9)
    if len(PC):
        q = d[d.pcms.notna()]; ax[2].scatter(q.pcms, q["stu_cell_cov>=0.7"] * 100, s=12, color=PAL[0], alpha=0.7)
        ax[2].set_xlabel("pathologists' % completely stained cells"); ax[2].set_ylabel("% cells ≥70% membrane-covered (model)")
        ax[2].set_title(f"ρ = {stats.spearmanr(q.pcms, q['stu_cell_cov>=0.7']).correlation:.2f}", fontsize=9)
    fig.suptitle(f"External dataset: {name} (frozen models, frozen stain calibration)", fontsize=10)
    savefig(fig, f"fig13_external_{name}")
    import zipfile
    zp = OUT / f"external_{name}_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in list(TAB.glob(f"external_{name}_*")) + list(FIG.glob(f"fig13_external_{name}*")):
            zf.write(f, f.relative_to(OUT.parent))
    print(f"\nexternal results: {zp}")
    return R, H1E, PC

_ext = run_external()
if _ext is not None:
    EXT_RESULTS = analyse_external(*_ext)
    section_time(f"external validation ({_ext[0]})")
