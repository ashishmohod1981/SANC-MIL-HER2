# %% [markdown]
# ## 1. Data discovery and validation
# This section looks for `train.csv`, `test.csv` and the image folders anywhere under `DATA_ROOT`, and accepts any common column spelling.
# It then checks the following:
# * every labelled file exists
# * the labels obey the scoring rules: 0 and 1+ are negative, 3+ is positive, 2+ can be either
# * the training filename prefix agrees with the IHC score
# * the image geometry is consistent
# * there are no exact duplicates within or across splits
#
# Near-duplicate detection runs later, using thumbnail hashes.

# %%
# ---------------- (debug only) synthetic HER2-like cores, so the full code path can be exercised without the real data ----
def make_synthetic_dataset(root, n_train=48, n_test=18, size=1024, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root); (root / "train").mkdir(parents=True, exist_ok=True); (root / "test").mkdir(parents=True, exist_ok=True)
    vH, vD = np.array([0.65, 0.70, 0.29]), np.array([0.27, 0.57, 0.78])
    from scipy.spatial import cKDTree
    def core(score):
        yy, xx = np.mgrid[:size, :size]
        r = size * 0.42; cy, cx = size / 2 + rng.normal(0, 10), size / 2 + rng.normal(0, 10)
        tissue = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        pts = rng.uniform(0, size, (int(size * size / 900), 2))
        lab = cKDTree(pts).query(np.stack([yy.ravel(), xx.ravel()], 1))[1].reshape(size, size)
        epi = ndi.gaussian_filter(rng.normal(0, 1, (size // 32, size // 32)), 1.5)
        epi = cv2.resize(epi, (size, size)) > 0.0
        bnd = find_boundaries(lab, mode="thick") & epi & tissue
        dist = np.linalg.norm(np.stack([yy.ravel(), xx.ravel()], 1) - pts[lab.ravel()], axis=1).reshape(size, size)
        nuc = (dist < 7) & tissue
        cH = 0.16 * tissue + 0.55 * nuc + 0.03 * rng.random((size, size)) * tissue
        amp = {0: 0.0, 1: 0.18, 2: 0.45, 3: 0.95}[score]
        keep = ndi.gaussian_filter(rng.random((size, size)), 6) > ({0: 1, 1: 0.55, 2: 0.45, 3: 0.3}[score])
        cD = amp * ndi.gaussian_filter((bnd & keep).astype(float), 0.8) * 1.8 + 0.02 * tissue * rng.random((size, size))
        od = cH[..., None] * vH + cD[..., None] * vD
        rgb = np.clip(245 * np.exp(-od) + rng.normal(0, 2, od.shape), 0, 255).astype(np.uint8)
        return rgb
    rows = {"train": [], "test": []}
    for split, n in [("train", n_train), ("test", n_test)]:
        for i in range(n):
            score = int(rng.choice([0, 0, 1, 2, 3, 3])) if split == "train" else int(rng.choice([0, 0, 2, 3]))
            status = 0 if score < 2 else (1 if score == 3 else int(rng.random() < 0.5))
            name = f"{score}_{i:04d}.jpg" if split == "train" else f"TMA_SP_{i:03d}.jpg"
            cv2.imwrite(str(root / split / name), cv2.cvtColor(core(score), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows[split].append({"filename": name, "ihc-score": score, "her2-status": status})
    pd.DataFrame(rows["train"]).to_csv(root / "train.csv", index=False)
    pd.DataFrame(rows["test"]).to_csv(root / "test.csv", index=False)

if RUN_MODE == "debug" and not (Path(CFG.DATA_ROOT).exists() and any(Path(CFG.DATA_ROOT).rglob("train.csv"))):
    print("DEBUG: generating synthetic cores at", CFG.DATA_ROOT)
    make_synthetic_dataset(CFG.DATA_ROOT)

# %%
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

ARCHIVE_EXT = (".zip", ".tar", ".tar.gz", ".tgz", ".rar", ".7z")
EXTRACT_DIR = Path("/tmp/her2_extracted" if IS_KAGGLE else "./extracted")   # /tmp: does not count against the 20 GB /kaggle/working limit

def discover(root):
    """Walk `root` FOLLOWING symlinks (Kaggle mounts dataset sub-folders as links) and index CSVs, images, archives."""
    csvs, images, archives, seen = {}, defaultdict(list), [], set()
    for dp, dn, fns in os.walk(root, followlinks=True):
        real = os.path.realpath(dp)
        if real in seen or real.startswith(str(OUT.resolve())) or real.startswith("/kaggle/working/her2_wsl"):
            dn[:] = []; continue
        seen.add(real)
        for fn in fns:
            p = Path(dp) / fn; low = fn.lower()
            if low == "train.csv" and "train" not in csvs: csvs["train"] = p
            elif low == "test.csv" and "test" not in csvs: csvs["test"] = p
            elif p.suffix.lower() in IMG_EXT:
                images[low].append(p); images["stem::" + Path(low).stem].append(p)
            elif low.endswith(ARCHIVE_EXT): archives.append(p)
    return csvs, images, archives

def extract_archives(archives, dest):
    """Unpack image archives that Kaggle did not auto-extract (.zip/.tar natively; .rar/.7z via installed tools)."""
    import zipfile, tarfile, shutil
    dest.mkdir(parents=True, exist_ok=True)
    for a in archives:
        out = dest / re.sub(r"\.(zip|tar|tar\.gz|tgz|rar|7z)$", "", a.name, flags=re.I)
        if out.exists() and any(out.rglob("*")): continue
        print(f"extracting {a} -> {out} (one-off; may take several minutes)")
        low = a.name.lower()
        try:
            if low.endswith(".zip"): zipfile.ZipFile(a).extractall(out)
            elif low.endswith((".tar", ".tar.gz", ".tgz")): tarfile.open(a).extractall(out)
            else:
                if shutil.which("unrar") is None and shutil.which("7z") is None:
                    subprocess.run("apt-get -qq install -y unrar p7zip-full > /dev/null 2>&1", shell=True)
                out.mkdir(parents=True, exist_ok=True)
                cmd = ["unrar", "x", "-o+", "-inul", str(a), str(out) + "/"] if shutil.which("unrar") and low.endswith(".rar") else ["7z", "x", "-y", f"-o{out}", str(a)]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"!! could not extract {a.name}: {e}. Extract it locally and upload the image folders instead.")

def standardize_labels(df, split):
    m = {_norm(c): c for c in df.columns}
    pick = lambda keys: next((m[k] for k in keys if k in m), None)
    fcol = pick(["filename", "file", "image", "imagename", "imageid", "path", "img", "name"])
    scol = pick(["ihcscore", "score", "ihc", "her2score"])
    tcol = pick(["her2status", "status", "her2", "label"])
    if fcol is None:
        raise ValueError(f"{split}.csv: no filename column in {list(df.columns)}")
    out = pd.DataFrame({"filename": df[fcol].astype(str).map(lambda s: Path(s).name)})
    out["ihc_score"] = pd.to_numeric(df[scol], errors="coerce") if scol else np.nan
    def st(v):
        if isinstance(v, str):
            v2 = v.strip().lower()
            if v2 in ("positive", "pos", "1", "true", "amplified"): return 1
            if v2 in ("negative", "neg", "0", "false", "not amplified"): return 0
            return np.nan
        return v
    out["her2_status"] = pd.to_numeric(df[tcol].map(st), errors="coerce") if tcol else np.nan
    out["split"] = split
    return out

csvs, IMG_INDEX, ARCHIVES = discover(CFG.DATA_ROOT)
TRAIN_CSV = Path(CFG.TRAIN_CSV) if CFG.TRAIN_CSV else csvs.get("train")
TEST_CSV = Path(CFG.TEST_CSV) if CFG.TEST_CSV else csvs.get("test")
assert TRAIN_CSV is not None and TEST_CSV is not None, f"train.csv/test.csv not found under {CFG.DATA_ROOT}"
def _match_rate():
    names = pd.concat([standardize_labels(pd.read_csv(p), s).filename for s, p in [("train", TRAIN_CSV), ("test", TEST_CSV)]])
    return np.mean([(n.lower() in IMG_INDEX) or (("stem::" + Path(n.lower()).stem) in IMG_INDEX) for n in names])
if _match_rate() < 0.5 and ARCHIVES:
    print(f"Only {_match_rate():.0%} of CSV filenames match loose images; found archive(s) {[a.name for a in ARCHIVES]} -> extracting")
    extract_archives(ARCHIVES, EXTRACT_DIR)
    _, extra, _ = discover(str(EXTRACT_DIR))
    for k, v in extra.items(): IMG_INDEX[k].extend(v)
n_img = sum(len(v) for k, v in IMG_INDEX.items() if not k.startswith("stem::"))
print("train csv:", TRAIN_CSV, "| test csv:", TEST_CSV, "| image files indexed:", n_img, f"| CSV match rate: {_match_rate():.1%}")

def resolve(fn, split):
    """Match a CSV filename (no path needed) to an indexed image: exact name (case-insensitive), then name without extension."""
    cands = IMG_INDEX.get(fn.lower(), []) or IMG_INDEX.get("stem::" + Path(fn.lower()).stem, [])
    if not cands: return None
    pref = [c for c in cands if split in str(c).lower()] or cands
    return str(max(pref, key=lambda c: os.path.getsize(c)))   # duplicate names: the largest (full-resolution) file wins

frames = []
for split, p in [("train", TRAIN_CSV), ("test", TEST_CSV)]:
    d = standardize_labels(pd.read_csv(p), split)
    d["path"] = [resolve(f, split) for f in d.filename]
    frames.append(d)
DF = pd.concat(frames, ignore_index=True)
if DF.path.notna().sum() == 0:
    tree = []
    for dp, dn, fns in os.walk(CFG.DATA_ROOT, followlinks=True):
        depth = dp[len(str(CFG.DATA_ROOT)):].count(os.sep)
        if depth <= 4: tree.append(f"{'  ' * depth}{os.path.basename(dp) or dp}/  ({len(fns)} files, e.g. {fns[:3]})")
    raise RuntimeError(
        f"None of the {len(DF)} filenames in the CSVs matched an image under {CFG.DATA_ROOT} "
        f"({n_img} images indexed; CSV example: {DF.filename.iloc[0]!r}; "
        f"indexed example: {next((k for k in IMG_INDEX if not k.startswith('stem::')), None)!r}; archives seen: {[a.name for a in ARCHIVES]}).\nFolder tree:\n" + "\n".join(tree[:60]) +
        "\nFix: attach the dataset that contains the train/test JPEGs (Add Input), or set CFG.DATA_ROOT to it.")
n_miss = int(DF.path.isna().sum())
if n_miss: print(f"WARNING: {n_miss} CSV rows have no matching image file (listed in data_validation.json)")
DF["image_id"] = DF.split + "__" + DF.filename.map(lambda s: Path(s).stem)

report = {}
report["n_rows"] = DF.groupby("split").size().to_dict()
report["missing_files"] = DF[DF.path.isna()].groupby("split").size().to_dict()
report["missing_labels"] = int(DF[["ihc_score", "her2_status"]].isna().any(axis=1).sum())
viol = DF[((DF.ihc_score <= 1) & (DF.her2_status == 1)) | ((DF.ihc_score == 3) & (DF.her2_status == 0))]
report["rule_violations(0/1+ positive or 3+ negative)"] = int(len(viol))
pref = DF[DF.split == "train"].filename.str.extract(r"^(\d)_")[0].astype(float)
report["train_prefix_vs_score_mismatch"] = int(((pref.notna()) & (pref != DF[DF.split == "train"].ihc_score)).sum())
DF = DF[DF.path.notna() & DF.ihc_score.notna() & DF.her2_status.notna()].reset_index(drop=True)
DF["ihc_score"] = DF.ihc_score.astype(int); DF["her2_status"] = DF.her2_status.astype(int)
DF["group"] = DF.ihc_score.astype(str)
DF.loc[DF.ihc_score == 2, "group"] = np.where(DF.loc[DF.ihc_score == 2, "her2_status"] == 1, "2+", "2-")

# geometry (header only) + quick content hash (size + first/last 1 MB) for exact duplicates
def header_and_hash(p):
    try:
        with Image.open(p) as im:
            w, h, fmt = im.size[0], im.size[1], im.format
    except Exception:
        return (np.nan, np.nan, None, None)
    sz = os.path.getsize(p); hsh = hashlib.md5(str(sz).encode())
    with open(p, "rb") as f:
        hsh.update(f.read(1 << 20)); f.seek(max(0, sz - (1 << 20))); hsh.update(f.read(1 << 20))
    return (w, h, fmt, hsh.hexdigest())
assert len(DF), "No labelled rows with a matching image remain - see the messages above."
meta = [header_and_hash(p) for p in tqdm(DF.path, desc="headers")]
DF["W"], DF["H"], DF["fmt"], DF["qhash"] = zip(*meta)
report["unreadable"] = int(DF.fmt.isna().sum())
DF = DF[DF.fmt.notna()].reset_index(drop=True)
_side = float(np.median(np.minimum(DF.W, DF.H)))
print(f"median core image size: {_side:.0f} px  ->  ≈{1200 / _side:.2f} µm/px if one image spans a 1.2 mm core")
if _side < CFG.MIN_CORE_PX and RUN_MODE != "debug" and not CFG.ALLOW_LOWRES:
    raise RuntimeError(
        f"The images are only ~{_side:.0f} px across (≈{1200 / _side:.1f} µm/px). At this scale a tumour cell is ~{12 * _side / 1200:.0f} px and a "
        "membrane is sub-pixel, so membrane/cell segmentation is physically impossible. Use the original full-resolution Zenodo cores "
        "(≈5468 px, 0.23 µm/px), or set CFG.ALLOW_LOWRES=True for a classification-only experiment.\nImages used came from e.g.: " + DF.path.iloc[0])
if abs(_side - 5468) / 5468 > 0.10 and _side >= CFG.MIN_CORE_PX:
    print(f"NOTE: cores are {_side:.0f} px, not the 5468 px in Pisula et al. If they were resized, set CFG.NATIVE_MPP = {0.23 * 5468 / _side:.3f}.")
report["image_sizes"] = DF.groupby(["W", "H"]).size().rename("n").reset_index().head(10).to_dict("records")
dup = DF[DF.duplicated("qhash", keep=False)]
report["exact_duplicate_files"] = int(len(dup))
report["exact_duplicates_across_splits"] = int(dup.groupby("qhash").split.nunique().gt(1).sum())
report["label_table"] = pd.crosstab(DF.split, DF.group).reindex(columns=GROUP_ORDER, fill_value=0).to_dict("index")
save_json(report, TAB / "data_validation.json")
print(json.dumps(report, indent=1, default=str))
display_tab = pd.crosstab(DF.split, DF.group).reindex(columns=GROUP_ORDER, fill_value=0)
print(display_tab)
if (display_tab.loc["test", "1"] if "test" in display_tab.index else 0) == 0:
    print("NOTE: the test set contains no IHC 1+ cores. Any 4-class test metric that reports a 1+ row is impossible.")
TRAIN = DF[DF.split == "train"].reset_index(drop=True)
TEST = DF[DF.split == "test"].reset_index(drop=True)

# %% [markdown]
# ## 2. Image I/O and the stain/morphology engine
# * **Reduced-resolution JPEG decoding.** PIL's `draft()` decodes the DCT directly at 1/2, 1/4 or 1/8 scale. A 5,468² core decodes at the
#   ≈20× work level about 4× faster, with no resampling artefacts.
# * **One dataset-level stain matrix.** It is estimated with Macenko on pooled pixels from a stratified sample and checked against Ruifrok's reference
#   vectors. Background illumination is estimated per image. **DAB is never min–max normalised per image.** That normalisation would make an IHC-0
#   core's faint background look like 3+ staining.
# * **Membrane-shape test on the DAB optical-density image.** It combines three checks:
#   1. *Ridge*: the multi-scale Hessian's dominant eigenvalue is negative and the structure is line-like.
#   2. *Locally prominent*: a white top-hat (radius 2 µm) shows the DAB rises above its own surround by ≥55%. Diffuse
#      cytoplasmic staining and solid blobs fail this; a membrane over weaker cytoplasm passes, even in dense 3+ tissue.
#   3. *Not nuclear*: it lies outside haematoxylin-**dominant** nuclei (H ≥ t_h and H ≥ DAB).
#
#   All three steps run on the GPU.

# %%
def read_rgb(path, ds):
    """Decode at 1/ds of native resolution. Uses JPEG DCT scaling when possible."""
    im = Image.open(path)
    W, H = im.size
    tw, th = max(1, W // ds), max(1, H // ds)
    if ds > 1 and im.format == "JPEG":
        im.draft("RGB", (tw, th))
    im = im.convert("RGB")
    if im.size != (tw, th):
        im = im.resize((tw, th), Image.BILINEAR)
    return np.asarray(im)

RUIFROK_H = np.array([0.650, 0.704, 0.286]); RUIFROK_D = np.array([0.268, 0.570, 0.776])
def _unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)
def complete_matrix(h, d):
    """Third (residual) stain = unit normal of the H-DAB plane. v1 bug fixed: taking abs() of the cross product made the
    residual almost parallel to haematoxylin (cos 0.99, condition number 15 vs 3), corrupting the H channel."""
    h, d = _unit(h), _unit(d); r = _unit(np.cross(h, d))
    return np.stack([h, d, r]).astype(np.float32)

def estimate_I0(thumb):
    """Per-channel background illumination from the brightest (non-tissue) pixels."""
    flat = thumb.reshape(-1, 3).astype(np.float32)
    bright = flat[flat.sum(1) > np.percentile(flat.sum(1), 90)]
    return np.clip(np.median(bright, 0), 180, 255) if len(bright) else np.array([255., 255., 255.])

def od_np(rgb, I0):
    return np.maximum(-np.log((rgb.astype(np.float32) + 1.0) / (np.asarray(I0, np.float32) + 1.0)), 0)

def tissue_from_thumb(thumb, I0):
    od = od_np(thumb, I0); om = od.max(-1)
    # glass is the brightest mode and covers >=10% of a TMA-core image; threshold sits a fixed OD margin above it.
    # (Otsu was rejected: strongly DAB-stained cores pull it upward and erase pale stroma -> class-dependent tissue masks.)
    thr = float(np.clip(np.percentile(om, 10) + 0.045, *CFG.OD_TISSUE_RANGE))
    t = (om > thr).astype(np.uint8)
    k = max(3, int(round(um2px(8, THUMB_MPP))) | 1); se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    t = cv2.morphologyEx(cv2.morphologyEx(t, cv2.MORPH_CLOSE, se), cv2.MORPH_OPEN, se)
    n, lab, st, _ = cv2.connectedComponentsWithStats(t, 8)
    min_area = (50.0 / THUMB_MPP) ** 2            # drop specks < 50x50 µm
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    t = keep[lab].astype(np.uint8)
    # core envelope = heavy closing + hole filling; rim = band inside the envelope boundary
    kc = max(3, int(round(um2px(120, THUMB_MPP))) | 1)
    env = cv2.morphologyEx(t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    env = ndi.binary_fill_holes(env).astype(np.uint8)
    dist = cv2.distanceTransform(env, cv2.DIST_L2, 5) * THUMB_MPP
    rim = (dist > 0) & (dist < CFG.RIM_UM)
    chroma = thumb.max(-1).astype(np.int16) - thumb.min(-1).astype(np.int16)
    dark = (od.sum(-1) > CFG.OD_DARK) & (chroma <= CFG.DARK_MAX_CHROMA)   # neutral-dark only: strong (brown) DAB is not an artefact
    return t.astype(bool), rim, dark, env.astype(bool)

def _gauss_1d(sigma, device):
    r = int(math.ceil(3.5 * sigma)); x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    g1 = -x / sigma ** 2 * g
    g2 = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    return g, g1, g2

def sep_conv(img, kx, ky):
    """img (1,1,H,W); kx filters along width, ky along height."""
    px, py = kx.numel() // 2, ky.numel() // 2
    out = F.conv2d(F.pad(img, (px, px, 0, 0), mode="reflect"), kx.view(1, 1, 1, -1))
    return F.conv2d(F.pad(out, (0, 0, py, py), mode="reflect"), ky.view(1, 1, -1, 1))

def hessian_ridge(x, sigmas_px):
    """Bright-ridge detector. Returns (scale-normalised magnitude σ²|λ_big|, line-likeness ∈[0,1]); zero where not a ridge."""
    best_m = torch.zeros_like(x); best_l = torch.zeros_like(x)
    for s in sigmas_px:
        g, g1, g2 = _gauss_1d(s, x.device)
        Ixx = sep_conv(x, g2, g) * s * s; Iyy = sep_conv(x, g, g2) * s * s; Ixy = sep_conv(x, g1, g1) * s * s
        mean = (Ixx + Iyy) / 2; dev = torch.sqrt(((Ixx - Iyy) / 2) ** 2 + Ixy ** 2)
        l1, l2 = mean + dev, mean - dev
        big = torch.where(l1.abs() > l2.abs(), l1, l2); small = torch.where(l1.abs() > l2.abs(), l2, l1)
        ridge = big < 0
        mag = torch.where(ridge, -big, torch.zeros_like(big))
        lin = torch.where(ridge, 1 - small.abs() / (big.abs() + 1e-6), torch.zeros_like(big))
        upd = mag > best_m
        best_m = torch.where(upd, mag, best_m); best_l = torch.where(upd, lin, best_l)
    return best_m, best_l

def dilate(m, r):
    if r < 1: return m
    k = 2 * int(r) + 1
    return F.max_pool2d(m.float(), k, 1, k // 2) > 0.5
def erode(m, r):
    if r < 1: return m
    k = 2 * int(r) + 1
    return ~(F.max_pool2d((~m).float(), k, 1, k // 2) > 0.5)

class StainEngine:
    def __init__(self, M, calib=None):
        self.M = torch.tensor(M, dtype=torch.float32, device=DEVICE)
        self.Minv = torch.linalg.pinv(self.M)
        self.cal = calib
        self.sig = [um2px(s) for s in CFG.RIDGE_SIGMAS_UM]

    @torch.no_grad()
    def maps(self, rgb, I0, tissue_thumb=None, rim_thumb=None, need_ridge=True):
        """rgb: HxWx3 uint8 at work level. Returns dict of (1,1,H,W) tensors."""
        H, W = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEVICE).float()
        I0t = torch.tensor(np.asarray(I0, np.float32), device=DEVICE)
        od = torch.clamp(-torch.log((x + 1) / (I0t + 1)), min=0)
        c = torch.clamp(od.reshape(-1, 3) @ self.Minv, min=0).reshape(H, W, 3)
        out = {"H": c[..., 0][None, None], "DAB": c[..., 1][None, None]}
        tis = od.max(-1).values[None, None] > CFG.OD_GAP
        if tissue_thumb is not None:
            tt = torch.from_numpy(tissue_thumb.astype(np.float32))[None, None].to(DEVICE)
            tis = tis & (F.interpolate(tt, size=(H, W), mode="nearest") > 0.5)
        out["tissue"] = tis
        if rim_thumb is not None:
            rt = torch.from_numpy(rim_thumb.astype(np.float32))[None, None].to(DEVICE)
            out["rim"] = (F.interpolate(rt, size=(H, W), mode="nearest") > 0.5) & tis
        else:
            out["rim"] = torch.zeros_like(tis)
        gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[None, None]
        out["gray"] = gray
        if not need_ridge:
            return out
        dab = out["DAB"]
        mag, lin = hessian_ridge(dab, self.sig)
        out["ridge_mag"], out["ridge_lin"] = mag, lin
        # white top-hat: DAB in excess of its ~4 um surround. Membranes rise above (weaker) cytoplasm; diffuse
        # cytoplasmic staining and solid blobs do not. (v1 used an opening-based "thick" test that erased 3+ membranes.)
        r = max(1, int(round(um2px(CFG.TOPHAT_R_UM)))); k = 2 * r + 1
        opened = F.max_pool2d(-F.max_pool2d(-dab, k, 1, r), k, 1, r)
        tophat = (dab - opened).clamp(min=0); out["tophat"] = tophat
        shape = (lin >= CFG.RIDGE_LIN_MIN) & (mag > 0) & (tophat >= CFG.TOPHAT_ABS) & (tophat >= CFG.TOPHAT_REL * dab) & tis
        g1 = _gauss_1d(1.0, DEVICE)[0]
        Hs, Ds = sep_conv(out["H"], g1, g1), sep_conv(dab, g1, g1)
        out["Hs"], out["Ds"] = Hs, Ds
        if self.cal is not None:
            nuc = (Hs >= self.cal["t_h"]) & (Hs >= Ds)          # haematoxylin-DOMINANT blobs only (dark DAB is not a nucleus)
            nuc = erode(dilate(nuc, 1), 1) & tis
            out["nuc"] = nuc
            mem_shape = shape & ~dilate(nuc, max(1, round(um2px(CFG.NUC_DILATE_UM))))
            out["mem_shape"] = mem_shape
            out["nonmem"] = (dab >= self.cal["t_weak"]) & tis & ~dilate(mem_shape, 1) & ~nuc   # cytoplasmic / blob DAB
            for kk in ("weak", "mod", "strong"):
                out[f"mem_{kk}"] = mem_shape & (dab >= self.cal[f"t_{kk}"])
        else:
            out["mem_shape"] = shape
        return out
