# %% [markdown]
# ## 3. Stain calibration (pass 0, stratified sample)
# 1. A Macenko stain matrix is estimated on pooled thumbnails, with up to `N_CALIB_PER_SCORE` cores per IHC score, and checked against Ruifrok.
# 2. The nuclear threshold `t_h` is the Otsu threshold of haematoxylin OD on tissue.
# 3. The DAB thresholds are **label-anchored**:
#    * `t_weak` is the 99th percentile of membrane-shaped, non-nuclear DAB OD in **IHC-0** cores, i.e. "rarely reached by staining in pathologist-negative cores".
#    * `t_bg` is the 90th percentile of the same values, the level below which a pixel is a confident negative.
#    * `t_mod` and `t_strong` are the boundaries of a 3-component Gaussian mixture fitted to log-DAB above `t_weak`.
#
# These thresholds are reproducible operating points. They are not a claim that they reproduce a pathologist's weak/moderate/strong grading.

# %%
CALIB_PATH = CACHE / "calibration.json"
def thumb_of(work):
    h, w = work.shape[:2]
    return cv2.resize(work, (max(1, w // CFG.THUMB_REL), max(1, h // CFG.THUMB_REL)), interpolation=cv2.INTER_AREA)

def macenko_global(thumbs_I0, per_img=20000, rng=np.random.default_rng(0)):
    pool = []
    for th, I0 in thumbs_I0:
        od = od_np(th, I0).reshape(-1, 3)
        od = od[(od.max(1) > 0.15) & (od.sum(1) < CFG.OD_DARK)]
        if len(od) > per_img: od = od[rng.choice(len(od), per_img, replace=False)]
        pool.append(od)
    od = np.concatenate(pool)
    _, V = np.linalg.eigh(np.cov(od.T)); V = V[:, [2, 1]]
    V = V * np.sign(V.sum(0) + 1e-12)
    phi = np.arctan2(od @ V[:, 1], od @ V[:, 0])
    a1, a2 = np.percentile(phi, 1), np.percentile(phi, 99)
    v1 = _unit(V @ [np.cos(a1), np.sin(a1)]); v2 = _unit(V @ [np.cos(a2), np.sin(a2)])
    v1, v2 = np.abs(v1), np.abs(v2)
    h, d = (v1, v2) if v1 @ RUIFROK_H > v2 @ RUIFROK_H else (v2, v1)
    info = {"cos_H_vs_ruifrok": float(h @ RUIFROK_H), "cos_DAB_vs_ruifrok": float(d @ RUIFROK_D)}
    if info["cos_H_vs_ruifrok"] < 0.90: h = RUIFROK_H; info["H_fallback"] = True
    if info["cos_DAB_vs_ruifrok"] < 0.85: d = RUIFROK_D; info["DAB_fallback"] = True
    return complete_matrix(h, d), info

def gmm_boundaries(logv, k=3, seed=0):
    g = GaussianMixture(k, random_state=seed, n_init=3).fit(logv.reshape(-1, 1))
    order = np.argsort(g.means_.ravel())
    grid = np.linspace(logv.min(), logv.max(), 4000).reshape(-1, 1)
    post = g.predict_proba(grid)[:, order]
    lab = post.argmax(1)
    bnds = [grid[np.argmax(lab >= j), 0] for j in range(1, k)]
    return bnds, g

if CALIB_PATH.exists():
    CALIB = json.load(open(CALIB_PATH)); print("loaded calibration", CALIB_PATH)
else:
    cal_df = pd.concat([TRAIN[TRAIN.ihc_score == s].sample(min(CFG.N_CALIB_PER_SCORE, (TRAIN.ihc_score == s).sum()),
                                                              random_state=CFG.SEED) for s in range(4)]).reset_index(drop=True)
    thumbs = []
    for p in tqdm(cal_df.path, desc="calib thumbs"):
        th = read_rgb(p, CFG.WORK_DS * CFG.THUMB_REL); thumbs.append((th, estimate_I0(th)))
    M, minfo = macenko_global(thumbs)
    eng0 = StainEngine(M, None)
    rng = np.random.default_rng(CFG.SEED)
    Hs, samples = [], []
    for (th, I0), (_, r) in tqdm(list(zip(thumbs, cal_df.iterrows())), desc="calib work-level"):
        work = read_rgb(r.path, CFG.WORK_DS)
        tis_t, rim_t, _, _ = tissue_from_thumb(th, I0)
        mp = eng0.maps(work, I0, tis_t, rim_t)
        tis = mp["tissue"][0, 0]; interior = tis & ~mp["rim"][0, 0]; shp = mp["mem_shape"][0, 0] & interior
        hv = mp["H"][0, 0][tis].cpu().numpy(); Hs.append(hv[rng.choice(len(hv), min(20000, len(hv)), replace=False)] if len(hv) else hv)
        dv = mp["DAB"][0, 0][shp].cpu().numpy(); hv2 = mp["Hs"][0, 0][shp].cpu().numpy(); dsv = mp["Ds"][0, 0][shp].cpu().numpy()
        sel = rng.choice(len(dv), min(20000, len(dv)), replace=False) if len(dv) else np.array([], int)
        samples.append((int(r.ihc_score), r.image_id, dv[sel], hv2[sel], dsv[sel]))
        del mp, work
    from skimage.filters import threshold_otsu
    t_h = float(threshold_otsu(np.concatenate(Hs)))
    by_score = {s: [] for s in range(4)}
    for s, iid, dv, hv, dsv in samples:
        by_score[s].append(dv[~((hv >= t_h) & (hv >= dsv))])     # drop haematoxylin-dominant (nuclear) pixels
    neg = np.concatenate(by_score[0]) if by_score[0] else np.array([0.05])
    t_weak = float(np.quantile(neg, CFG.Q_NEG)); t_bg = float(np.quantile(neg, CFG.Q_BG))
    above = np.concatenate([v[v > t_weak] for s in (1, 2, 3) for v in by_score[s]] or [np.array([])])
    try:
        assert len(above) > 500
        (b1, b2), _ = gmm_boundaries(np.log(above))
        t_mod, t_strong = float(np.exp(b1)), float(np.exp(b2))
        assert t_weak < t_mod < t_strong
        how = "gmm3"
    except Exception:
        s3 = np.concatenate([v[v > t_weak] for v in by_score[3]] or [np.array([t_weak * 1.5, t_weak * 3])])
        t_mod, t_strong = float(np.quantile(s3, 1 / 3)), float(np.quantile(s3, 2 / 3)); how = "quantile_fallback"
        t_mod = max(t_mod, t_weak * 1.05); t_strong = max(t_strong, t_mod * 1.05)
    CALIB = dict(M=M.tolist(), macenko=minfo, t_h=t_h, t_bg=t_bg, t_weak=t_weak, t_mod=t_mod, t_strong=t_strong,
                 band_method=how, n_calib_images=len(cal_df), q_neg=CFG.Q_NEG)
    save_json(CALIB, CALIB_PATH)
    # figure: membrane-shaped DAB OD distribution by IHC score with operating points
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for s in range(4):
        v = np.concatenate(by_score[s]) if by_score[s] else np.array([])
        v = v[v > 1e-3]
        if len(v): ax.hist(np.log10(v), bins=120, density=True, histtype="step", lw=1.6, color=GROUP_COL[str(s) if s != 2 else "2+"], label=f"IHC {s} (n={len(by_score[s])} cores)")
    for k, ls in [("t_bg", ":"), ("t_weak", "--"), ("t_mod", "-."), ("t_strong", "-")]:
        ax.axvline(np.log10(CALIB[k]), color="#52514e", ls=ls, lw=1); ax.text(np.log10(CALIB[k]), ax.get_ylim()[1] * 0.92, k, rotation=90, fontsize=7, ha="right", color="#52514e")
    ax.set_xlabel("log10 DAB optical density (membrane-shaped, non-nuclear pixels)"); ax.set_ylabel("density"); ax.legend(fontsize=7)
    ax.set_title("Label-anchored DAB operating points")
    savefig(fig, "fig02_dab_calibration")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in CALIB.items() if k != "M"})
ENGINE = StainEngine(np.array(CALIB["M"]), CALIB)

# %% [markdown]
# ## 4. Pass 1: one decode per core yields tissue, QC, stain features and 20× tile embeddings
# For every core (train and test) this pass stores the following, and can resume from a partial run:
# * the tissue mask, the rim band and dark artefacts at about 1.8 µm/px
# * 224-px tiles (about 103 µm) at about 0.46 µm/px, kept when ≥50% of the tile is tissue
# * one encoder embedding per tile
# * 14 interpretable stain and morphology features per tile
# * QC metrics
# * low-resolution DAB and membrane maps for figures

# %%
HAND_NAMES = ["tissue_frac", "dab_mean", "dab_std", "dab_ge_weak", "dab_ge_mod", "dab_ge_strong", "mem_weak", "mem_mod",
              "mem_strong", "nonmem_dab", "nuclear_dab", "nuc_frac", "focus", "rim_frac"]
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class TileEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.kind, name = spec.split(":", 1)
        self.pretrained = True
        if self.kind == "timm":
            try:
                self.m = timm.create_model(name, pretrained=True, features_only=True, out_indices=(CFG.ENCODER_OUT_INDEX,))
            except Exception as e:
                print(f"!! WARNING: could not download pretrained '{name}' ({type(e).__name__}). Using RANDOM weights. "
                      "Enable Internet in Kaggle settings or attach the weights; results will be meaningless otherwise.")
                self.m = timm.create_model(name, pretrained=False, features_only=True, out_indices=(CFG.ENCODER_OUT_INDEX,))
                self.pretrained = False
            self.dim = self.m.feature_info.channels()[-1]
            pc = getattr(self.m, "pretrained_cfg", None) or {}
            mean, std = pc.get("mean", IMNET_MEAN), pc.get("std", IMNET_STD)
        else:
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained(name); self.dim = self.m.config.hidden_size
            mean, std = IMNET_MEAN, IMNET_STD
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.kind == "timm":
            return self.m(x)[-1].mean((2, 3))
        return self.m(pixel_values=x).last_hidden_state[:, 0]

ALL = pd.concat([TRAIN, TEST], ignore_index=True)
FEAT_DIR, MAP_DIR, THUMB_DIR = CACHE / "feats", CACHE / "maps", CACHE / "thumbs"
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)

def dhash(gray, n=16):
    g = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return "".join("%x" % int("".join(map(str, bits[i:i + 4].astype(int))), 2) for i in range(0, len(bits), 4))

class DecodeDS(torch.utils.data.Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        try: return i, read_rgb(self.df.path[i], CFG.WORK_DS)
        except Exception: return i, None

@torch.no_grad()
def featurize(row, work, enc, enc2=None):
    iid = row.image_id; T = CFG.TILE
    H, W = work.shape[:2]
    thumb = thumb_of(work); I0 = estimate_I0(thumb)
    tis_t, rim_t, dark_t, env_t = tissue_from_thumb(thumb, I0)
    mp = ENGINE.maps(work, I0, tis_t, rim_t)
    ny, nx = H // T, W // T
    cut = lambda m: m.float()[..., :ny * T, :nx * T]
    pool = lambda m: F.avg_pool2d(cut(m), T, T)[0, 0]
    tis = mp["tissue"]; tf = pool(tis); tisN = tf.clamp(min=1e-6)
    frac = lambda m: pool(m & tis) / tisN
    dab = mp["DAB"] * tis
    dmean = pool(dab) / tisN; dsq = pool(dab * dab) / tisN
    lap = F.conv2d(F.pad(mp["gray"], (1, 1, 1, 1), mode="reflect"), LAP)
    focus = pool(lap * lap) - pool(lap) ** 2
    feats = torch.stack([tf, dmean, torch.sqrt((dsq - dmean ** 2).clamp(min=0)),
                         frac(mp["DAB"] >= CALIB["t_weak"]), frac(mp["DAB"] >= CALIB["t_mod"]), frac(mp["DAB"] >= CALIB["t_strong"]),
                         frac(mp["mem_weak"]), frac(mp["mem_mod"]), frac(mp["mem_strong"]), frac(mp["nonmem"]),
                         frac(mp["nuc"] & (mp["DAB"] >= CALIB["t_weak"])), frac(mp["nuc"]), focus, pool(mp["rim"])], -1)
    keep = torch.nonzero(tf >= CFG.MIN_TILE_TISSUE).cpu().numpy()
    if len(keep) > CFG.MAX_TILES_PER_IMAGE:
        rs = np.random.default_rng(int(hashlib.md5(iid.encode()).hexdigest()[:8], 16))
        keep = keep[np.sort(rs.choice(len(keep), CFG.MAX_TILES_PER_IMAGE, replace=False))]
    hand = feats[keep[:, 0], keep[:, 1]].cpu().numpy().astype(np.float32) if len(keep) else np.zeros((0, len(HAND_NAMES)), np.float32)
    coords = (keep * T).astype(np.int32)
    xg = torch.from_numpy(work).to(DEVICE).permute(2, 0, 1)
    embs, embs2 = [], []
    for b in range(0, len(coords), CFG.ENC_BATCH):
        cb = coords[b:b + CFG.ENC_BATCH]
        batch = torch.stack([xg[:, y:y + T, x:x + T] for y, x in cb]).float() / 255.0
        with autocast():
            embs.append(gpu_safe(enc, batch).float().cpu())
            if enc2 is not None: embs2.append(gpu_safe(enc2, batch).float().cpu())
    emb = torch.cat(embs).numpy().astype(np.float16) if embs else np.zeros((0, enc.dim), np.float16)
    extra = {}
    if enc2 is not None:
        extra["emb2"] = torch.cat(embs2).numpy().astype(np.float16) if embs2 else np.zeros((0, enc2.dim), np.float16)
    # ---------- QC ----------
    tissue_px = max(int(tis_t.sum()), 1)
    full_lap = lap[0, 0][tis[0, 0]]
    dabpos = (mp["DAB"] >= CALIB["t_weak"]) & tis
    if int(dabpos.sum()) > 500:
        x = torch.from_numpy(work).to(DEVICE).float(); I0t = torch.tensor(I0, device=DEVICE, dtype=torch.float32)
        odv = torch.clamp(-torch.log((x + 1) / (I0t + 1)), min=0)[dabpos[0, 0]].mean(0).cpu().numpy()
        drift = float(_unit(odv) @ np.array(CALIB["M"])[1])
    else:
        drift = np.nan
    interior = tis & ~mp["rim"]
    mi = float((mp["mem_weak"] & interior).sum() / interior.sum().clamp(min=1))
    mr = float((mp["mem_weak"] & mp["rim"]).sum() / mp["rim"].sum().clamp(min=1))
    qc = dict(image_id=iid, n_tiles=int(len(coords)), tissue_mm2=float(tis_t.sum() * THUMB_MPP ** 2 / 1e6),
              dark_frac=float((dark_t & tis_t).sum() / tissue_px), focus_med=float(np.median(hand[:, 12])) if len(hand) else np.nan,
              global_lapvar=float(full_lap.var()) if full_lap.numel() > 10 else np.nan, I0=float(np.mean(I0)),
              dab_hue_cos=drift, mem_weak_interior=mi, mem_weak_rim=mr,
              dab_mean_core=float((mp["DAB"] * tis).sum() / tis.sum().clamp(min=1)), dhash=dhash(cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)),
              work_h=H, work_w=W)

    R = CFG.THUMB_REL
    lr = lambda m: F.avg_pool2d(m.float()[..., :H // R * R, :W // R * R], R, R)[0, 0].cpu().numpy().astype(np.float16)
    atomic_npz(MAP_DIR / f"{iid}.npz", compressed=True, dab=lr(mp["DAB"] * tis), mem_weak=lr(mp["mem_weak"]), mem_mod=lr(mp["mem_mod"]),
               tissue=tis_t, rim=rim_t, I0=I0)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    atomic_write(THUMB_DIR / f"{iid}.jpg", lambda f: f.write(jpg.tobytes()))
    # the features file is written LAST: its presence marks the core as complete (maps and thumbnail already on disk)
    atomic_npz(FEAT_DIR / f"{iid}.npz", coords=coords, emb=emb, hand=hand, qc=json.dumps(qc), **extra)
    return qc

def _core_done(iid):
    """A core is complete only if all three files exist and the feature file opens (guards against truncated files)."""
    f = FEAT_DIR / f"{iid}.npz"
    if not (f.exists() and (MAP_DIR / f"{iid}.npz").exists() and (THUMB_DIR / f"{iid}.jpg").exists()): return False
    try:
        with np.load(f) as z: return {"coords", "emb", "hand", "qc"} <= set(z.files)
    except Exception:
        f.unlink(missing_ok=True); return False
todo = ALL[[not _core_done(i) for i in ALL.image_id]].reset_index(drop=True)
print(f"pass 1: {len(todo)} / {len(ALL)} cores to featurize")
ENC_INFO_PATH = CACHE / "encoder_info.json"
if len(todo):
    ENC = TileEncoder(CFG.ENCODER).to(DEVICE).eval()
    ENC2 = None
    if CFG.ENCODER2:
        try:
            ENC2 = TileEncoder(CFG.ENCODER2).to(DEVICE).eval()
            print(f"encoder ablation: {CFG.ENCODER2} ({ENC2.dim}-d) extracted in the same pass")
        except Exception as e:
            print(f"!! {CFG.ENCODER2} unavailable ({type(e).__name__}: {str(e)[:120]}); encoder ablation skipped")
    if not ENC_INFO_PATH.exists():
        save_json(dict(encoder=CFG.ENCODER, pretrained=ENC.pretrained, dim=ENC.dim,
                       encoder2=CFG.ENCODER2 if ENC2 is not None else None, dim2=ENC2.dim if ENC2 is not None else None), ENC_INFO_PATH)
    dl = torch.utils.data.DataLoader(DecodeDS(todo), batch_size=1, num_workers=CFG.NUM_WORKERS,
                                     collate_fn=lambda b: b[0], prefetch_factor=2 if CFG.NUM_WORKERS else None)
    t0 = time.time()
    ENC2_USE = ENC2 if (ENC2 is not None and json.load(open(ENC_INFO_PATH)).get("encoder2")) else None
    for i, work in tqdm(dl, total=len(todo), desc="featurize"):
        if work is None:
            print("unreadable:", todo.path[i]); continue
        featurize(todo.iloc[i], work, ENC, ENC2_USE)
    print(f"pass 1 done in {(time.time() - t0) / 60:.1f} min")
    del ENC, ENC2; gc.collect(); torch.cuda.empty_cache()
ENC_INFO = json.load(open(ENC_INFO_PATH)) if ENC_INFO_PATH.exists() else dict(pretrained=True, encoder2=None)
ENCODER_PRETRAINED = ENC_INFO["pretrained"]
CFG.ENCODER2 = ENC_INFO.get("encoder2")     # the encoder ablation runs only if every core carries the second embedding
section_time("pass 1 featurisation")

# %%
# ---------------- membrane-detector QA figure (for pathologist review of the morphological prior) ----------------
def qa_membrane_figure(n=3):
    picks = []
    for s in [3, 2, 0]:
        sub = TRAIN[TRAIN.ihc_score == s]
        if len(sub): picks.append(sub.sample(1, random_state=CFG.SEED).iloc[0])
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.9 * len(picks)))
    axes = np.atleast_2d(axes)
    for ax_row, r in zip(axes, picks):
        work = read_rgb(r.path, CFG.WORK_DS); th = thumb_of(work); I0 = estimate_I0(th)
        tis_t, rim_t, _, _ = tissue_from_thumb(th, I0)
        mp = ENGINE.maps(work, I0, tis_t, rim_t)
        d = F.avg_pool2d(mp["DAB"] * mp["tissue"], CFG.SEG_CROP, CFG.SEG_CROP // 2)[0, 0]
        yy, xx = np.unravel_index(int(d.argmax()), d.shape); S = CFG.SEG_CROP
        y0, x0 = min(yy * S // 2, work.shape[0] - S), min(xx * S // 2, work.shape[1] - S)
        sl = (slice(y0, y0 + S), slice(x0, x0 + S))
        g = lambda k: mp[k][0, 0].cpu().numpy()[sl]
        ax_row[0].imshow(work[sl]); ax_row[0].set_title(f"{r.image_id[:22]}  IHC {r.ihc_score}")
        ax_row[1].imshow(g("DAB"), cmap="magma", vmin=0, vmax=CALIB["t_strong"] * 1.5); ax_row[1].set_title("DAB OD (absolute)")
        ov = work[sl].copy()
        ov[g("mem_weak")] = (42, 120, 214); ov[g("mem_mod")] = (235, 104, 52); ov[g("mem_strong")] = (227, 73, 72)
        ax_row[2].imshow(ov); ax_row[2].set_title("membrane DAB\n(blue weak / orange mod / red strong)", fontsize=8)
        ov2 = work[sl].copy(); ov2[g("nonmem")] = (237, 161, 0); ov2[g("nuc")] = (98, 80, 214)
        ax_row[3].imshow(ov2); ax_row[3].set_title("non-membranous DAB (amber)\n/ H-dominant nuclei (violet)", fontsize=8)
        for a in ax_row: a.axis("off")
    fig.suptitle("Morphological membrane prior: an unverified hypothesis for expert review", y=1.0)
    savefig(fig, "fig03_membrane_prior_QA")
if not (FIG / "fig03_membrane_prior_QA.png").exists(): qa_membrane_figure()
