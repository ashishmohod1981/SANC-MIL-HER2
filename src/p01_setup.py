# %% [markdown]
# # Weakly-supervised HER2 localisation in gastro-oesophageal adenocarcinoma TMA cores, without pixel masks
#
# **Dataset.** Pisula *et al.*, *Br J Cancer* 2023 (Zenodo 10.5281/zenodo.7031868, CC-BY 4.0). It has 1,598 training cores: a multi-spot TMA with
# 8 cores per tumour for 165 cases, and a single-spot TMA with 428 cases. The test set is 307 single-spot cores from **later** patients.
# Each core has two labels: an IHC score (0–3) and the HER2 status (score 2 was resolved by FISH). The test set has **no IHC 1+ cores**.
# There are **no patient identifiers and no pixel masks**.
#
# ## What the labels can and cannot teach (decided *before* any architecture choice)
# | Target | Supervision available | How this notebook handles it | Status of the output |
# |---|---|---|---|
# | Core HER2 status / IHC score | Direct (image level) | Multiple-instance learning (MIL) on 20× tiles, evaluated on the temporally independent test set | **Validated** (classification metrics with CIs) |
# | *Where* HER2-positive evidence lies | Indirect. Positive cores contain some positive tiles. IHC 0/1+ cores contain **no** 3+-type evidence, which is an exact negative constraint | Instance-level positivity, with the negative-bag constraint and a top-k positive constraint | Partially checkable: the false-positive rate on IHC-0 cores is a *real* error measure. Everything else is a proxy |
# | DAB chromogen amount | Physics (Beer–Lambert colour deconvolution) | Absolute optical density with **one** dataset-level stain matrix and label-anchored thresholds | Measurement, not a prediction |
# | *Membranous* versus cytoplasmic or nuclear DAB | **None.** This is a morphological judgement | Hessian line-likeness, a white top-hat local-prominence test, and haematoxylin-dominant nuclear exclusion, distilled into a CNN student trained with ignore regions and confidence weights | **Unverified hypothesis** until expert masks exist |
# | Tumour versus stroma | **None.** HER2 labels say nothing about tissue type | Unsupervised tile clusters plus a gallery that a pathologist names in about 10 minutes | Unverified until the clusters are named |
# | Cells and membrane completeness | **None** | Nuclei from haematoxylin, expanded into territories, then the fraction of each territory boundary that is DAB-membranous | Approximate morphometry and a proxy only |
#
# **Final rule enforced by the code:** with no reference masks, the notebook never prints Dice or IoU as accuracy. Agreement with pseudo-labels is reported
# under that name. Section 17 computes true Dice, IoU, HD95 and boundary F1 automatically once expert masks are supplied.
#
# ### Pathology conventions used
# * Gastric/GEA HER2 scoring follows Hofmann/Rüschoff. Membranous reactivity can be **complete, basolateral or lateral (U-shaped)**, so circumferential staining is *not* required.
#   Only membranous staining counts. Cytoplasmic and nuclear DAB are excluded from "membrane" by design.
# * IHC 2+ is **equivocal**. Its status comes from FISH. The pipeline reports a status probability for 2+ cores but always routes them to "ISH required".
# * The periphery of a TMA core is prone to edge artefact. A rim band is measured separately and ignored in the pseudo-labels.
# * The weak/moderate/strong DAB bands below are *data-derived operating points*, not validated equivalents of a pathologist's intensity grading.

# %% [markdown]
# ## Version 2: changes driven by the audit of the v1 run (full run on 1,905 cores, Tesla T4)
# | # | v1 finding (evidence) | v2 change |
# |---|---|---|
# | 1 | **Bug.** The residual stain vector was abs(H×DAB), nearly parallel to H (cos 0.988; condition number 15). This inflated or zeroed the haematoxylin channel. | Correct H×DAB normal (condition number 3). All features are recomputed in a fresh cache (`her2_wsl_v2`). |
# | 2 | **Bug.** The opening-based "thick DAB" test removed 30% of pixels in 3+ crops, so pseudo-labels taught that strong 3+ membranes are *not* membrane. IHC 3 cores showed less membrane than 2+. | A white top-hat local-prominence test plus a stricter line test. On 20 training crops, 3+ membrane coverage rose about 6× and 3+/0 contrast rose about 3×. |
# | 3 | Dark DAB was read as nuclei, because nuclei were defined as "H ≥ t_h" only. | Nuclei must be haematoxylin-**dominant** (H ≥ t_h and H ≥ DAB). |
# | 4 | The dark-artefact QC flag fired on 25% of test IHC 3 cores, because strong DAB looks "dark". | Dark artefact = very dark **and** near-neutral colour. |
# | 5 | 5 test cores with fewer than 8 tissue tiles were silently dropped from evaluation. | Test cores are never dropped. They are kept, flagged low-tissue, and routed to "refer". |
# | 6 | ISH routing with a fixed P(IHC2) ≥ 0.3 caught only 40% of IHC 2 test cores. | Threshold tuned on OOF predictions for 80% sensitivity. Test sensitivity and specificity are reported. |
# | 7 | Cell coverage did not separate 2+ from 3+, because territories were fixed 4 µm expansions unrelated to membranes. | Membrane-guided watershed: nuclei are the seeds and the membrane map is the barrier. |
# | 8 | Claims rested on single training runs. | Repeated CV seeds (`N_SEEDS`) for the key variants, reported as mean ± SD. |
# | 9 | There was no CAM baseline (the approach of the original draft) and no pathology-domain encoder. | Added a Grad-CAM thumbnail CNN baseline, scored with identical metrics, and a Phikon (pathology foundation model) encoder ablation from the same pass. |
# | 10 | Segmentation training froze twice in forked loader workers. The best epoch was not kept. | Single-process loading, per-epoch checkpoint and resume, and best-epoch selection. |
# | 11 | A localisation sensitivity check was missing. | IHC 3+ core detection and tile AUROC (tiles from IHC 3 cores versus IHC 0 cores) added alongside the IHC 0 false-positive rate. |
#
# The v1 architecture finding still stands. MSA-U-Net-TR did not beat a plain U-Net (same-budget high-confidence agreement 0.84 vs 0.87). Both are
# kept as a reported ablation, and no architectural novelty is claimed.

# %% [markdown]
# ## Version 2.1: restartability and additional evidence
# **Restart from where it stopped.** Every expensive loop now saves each finished item at once.
# * **How items are saved.** Most loops write each finished item to an append-only log (`cache/logs/*.jsonl`) and flush it to disk; feature extraction writes each core's `npz` file. Re-running a cell skips finished items and continues with the rest.
# * **What is covered.** Feature extraction (per core), MIL runs (per variant, fold set and seed), Grad-CAM (per fold), pseudo-label crops, core and cell features, and whole-core test segmentation (per core). Segmentation training resumes per epoch.
# * **Crash safety.** All cache files are written to a temporary name and then renamed, so a crash can never leave a half-written file that looks finished.
# * **Same result after a restart.** Random draws are seeded per core, so resumed output does not depend on where the run stopped.
# * **Within one session:** just run all cells again.
# * **In a new Kaggle session:** use *Add Input*, pick the stopped notebook's output, then *Run All*. The first cell finds `her2_wsl_v2` and copies it into `/kaggle/working`, and the run continues from there. To point at it explicitly, set the `HER2_RESUME_FROM` environment variable.
#
# **New evidence for the manuscript.** Each item below is computed. None of it is asserted.
# | Item | Output |
# |---|---|
# | H1: paired per-core Wilcoxon and bootstrap tests of IHC-0 false-positive tiles, per seed, on the test set **and** on out-of-fold training cores; CI for the FP rate | `tables/h1_negative_constraint_tests.csv`, `fig11` |
# | H3: single-feature AUROC for 3+ vs 2+ and for 2+/FISH+ vs 2+/FISH−; direction fixed on train; stratified bootstrap CI; Holm correction | `tables/h3_feature_separation_3v2_and_fish.csv`, `fig12` |
# | Holm-corrected p-values for every paired AUROC comparison with the proposed model | `tables/paired_delta_auc.csv` |
# | Convergence curves (MIL per fold; segmentation per epoch, selected epoch marked) | `figS2_convergence` |
# | Computational cost: parameters, per-core inference time, wall-clock per section | `tables/compute_*.csv` |
# | Claims ledger wording gated on pre-stated criteria (e.g. H1 is called "supported" only if every seed's CI lies below 0) | `RESULTS_SUMMARY.md` |

# %%
# ============================== 0. ENVIRONMENT & CONFIG ==============================
import os, sys, json, math, time, random, glob, hashlib, warnings, gc, re, pickle, contextlib, subprocess
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
                             cohen_kappa_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
try:
    import timm
except ImportError:
    _pip("timm"); import timm
try:
    import skimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed, expand_labels, find_boundaries
except ImportError:
    _pip("scikit-image"); import skimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed, expand_labels, find_boundaries
from tqdm.auto import tqdm
# Keep scikit-learn / NumPy single-threaded for the whole notebook. After PyTorch's DataLoader workers have forked,
# two OpenMP runtimes can dead-lock scikit-learn on Kaggle (seen in k-means). GPU work is unaffected; CPU steps here are small.
from threadpoolctl import threadpool_limits
_THREADS = threadpool_limits(limits=1)

RUN_MODE = os.environ.get("HER2_RUN_MODE", "full").lower()   # 'full' | 'fast' | 'debug'
IS_KAGGLE = Path("/kaggle/input").exists()

class CFG:
    # ---------- paths (None = auto-discover under DATA_ROOT) ----------
    # Your full-resolution Zenodo images (training_dataset/, testing_dataset/, train.csv, test.csv).
    # Pointing at this folder (not all of /kaggle/input) stops other attached datasets with the same filenames being used.
    DATA_ROOT      = os.environ.get("HER2_DATA_ROOT", next((p for p in ["/kaggle/input/datasets/ashishpandharimohod/her2dataset",
                                                               "/kaggle/input/her2dataset"] if Path(p).exists()),
                                                      "/kaggle/input" if IS_KAGGLE else "./data"))
    MIN_CORE_PX    = 2000      # cores smaller than this cannot resolve membranes/cells -> stop with an explanation
    ALLOW_LOWRES   = False     # set True only for a classification-only experiment on downsampled images
    TRAIN_CSV      = None
    TEST_CSV       = None
    REF_MASK_DIR   = os.environ.get("HER2_REF_MASK_DIR") or None   # folder of expert masks (PNG, same stem as image or ROI) -> enables true Dice/IoU
    CLUSTER_NAMING_CSV = None  # filled cluster-naming sheet from a pathologist (Section 9) -> tumour-candidate maps
    OUT            = Path(os.environ.get("HER2_OUT", "/kaggle/working/her2_wsl_v2" if IS_KAGGLE else "./her2_wsl_out"))   # v2: fresh cache (stain matrix fixed)
    SEED           = 2024
    NUM_WORKERS    = 4

    # ---------- physical scale ----------
    NATIVE_MPP     = 0.23      # NanoZoomer S360 @40x (µm/px). Everything physical is derived from this.
    WORK_DS        = 2         # work level = native/2 ≈ 0.46 µm/px (≈20x): membranes are 2–4 px wide
    THUMB_REL      = 4         # thumbnail = work/4 ≈ 1.84 µm/px
    TILE           = 224       # MIL tile at work level ≈ 103 µm
    MIN_TILE_TISSUE= 0.5
    MAX_TILES_PER_IMAGE = 400
    MIN_TILES_KEEP = 8         # TRAIN cores with fewer tissue tiles are excluded; TEST cores are never dropped (>=1 tile kept, flagged)

    # ---------- stain / morphology ----------
    OD_TISSUE_RANGE= (0.05, 0.15)  # per-core Otsu threshold on max-channel OD, clipped to this range (tissue vs glass)
    OD_GAP         = 0.04      # at work level, pixels below this OD inside tissue are lumina/gaps
    OD_DARK        = 2.2       # summed OD above which a pixel may be a dark artefact ...
    DARK_MAX_CHROMA= 30        # ... but only if it is also near-neutral (max-min RGB <= this); strong brown DAB is NOT an artefact
    RIM_UM         = 60.0      # core periphery band (edge-artefact zone)
    RIDGE_SIGMAS_UM= (0.5, 0.75, 1.0)   # membrane half-widths probed by the Hessian
    RIDGE_LIN_MIN  = 0.75      # line-likeness (1-|λs|/|λb|) of a membrane ridge
    TOPHAT_R_UM    = 2.0       # white top-hat radius: membranes are thinner than ~4 µm
    TOPHAT_REL     = 0.55      # membrane DAB must exceed its ~4 µm surround by >= this fraction of its own OD
    TOPHAT_ABS     = 0.03      # ... and by >= this absolute OD
    NUC_DILATE_UM  = 0.8       # exclusion ring around haematoxylin-dominant nuclei
    # (v2) settings chosen by visual review of 20 TRAINING crops (4 per IHC group); see fig03. Unverified until expert review.
    Q_NEG          = 0.99      # t_weak = this quantile of membrane-shaped DAB OD in IHC-0 cores
    Q_BG           = 0.90      # t_bg   = this quantile (confident-negative level)
    N_CALIB_PER_SCORE = 40

    # ---------- encoder ----------
    ENCODER        = "timm:resnet50.tv_in1k"   # truncated after layer3 (CLAM convention, 1024-d)
    ENCODER_OUT_INDEX = 3
    ENCODER2       = "hf:owkin/phikon"         # pathology foundation model (ViT-B, open weights) for the encoder ablation; None to skip
    N_SEEDS        = 3         # repeated CV seeds for the key MIL variants (variability / CIs)
    RUN_GRADCAM    = True      # thumbnail-CNN + Grad-CAM baseline (the approach of the original draft / Pisula et al. ResNet)
    CAM_ARCH       = "resnet18.tv_in1k"
    CAM_EPOCHS     = 10
    CAM_BS         = 16
    ISH_ROUTE_TARGET_SENS = 0.80   # P(IHC2) routing threshold chosen on OOF to reach this sensitivity for IHC2 cores
    ENC_BATCH      = 128

    # ---------- MIL ----------
    N_FOLDS        = 5
    MIL_EPOCHS     = 40
    MIL_BS         = 16
    MIL_LR         = 2e-4
    MIL_WD         = 1e-3
    MIL_MAX_TILES_TRAIN = 256
    MIL_TILE_DROP  = 0.10
    TOPK_FRAC      = 0.05
    W_ORD, W_NEG, W_POS, W_CLAM = 0.5, 0.5, 0.5, 0.3
    MC_SAMPLES     = 10
    EXPECTED_PATIENTS = 593    # 165 multi-spot + 428 single-spot cases (paper)
    EXPECTED_CORES = 1748      # 165*8 + 428 before the authors discarded corrupted images

    # ---------- tissue clusters ----------
    N_TISSUE_CLUSTERS = 12

    # ---------- pseudo-labels & segmentation student ----------
    SEG_CROP       = 512
    CROPS_PER_IMAGE= 8
    SEG_ENCODER    = "resnet34.tv_in1k"
    SEG_EPOCHS     = 10
    SEG_ABL_EPOCHS = 4
    SEG_BS         = 8
    SEG_LR         = 3e-4
    EV_P_POS, EV_P_NEG, EV_STD_MAX = 0.80, 0.20, 0.15
    W_EV           = 0.5
    TTA            = 4
    MIN_MEM_OBJ_PX = 12
    FEAT_CROPS_PER_CORE = 12   # crops per core for core-level / cell-level features (same protocol train & test)
    CELL_NUC_MIN_UM= 3.0
    CELL_MAX_UM    = 8.0       # (v2) max cell radius from its nucleus for the membrane-guided watershed
    SAVE_FULLRES_MASKS = True

    # ---------- evaluation ----------
    N_BOOT         = 2000
    ABSTAIN_COVERAGE = 0.90
    N_ANNOT_ROIS_PER_CORE = 3

if RUN_MODE == "fast":
    CFG.N_SEEDS, CFG.CAM_EPOCHS = 2, 6
    CFG.MIL_EPOCHS, CFG.SEG_EPOCHS, CFG.SEG_ABL_EPOCHS = 25, 6, 2
    CFG.CROPS_PER_IMAGE, CFG.FEAT_CROPS_PER_CORE, CFG.MC_SAMPLES, CFG.N_BOOT = 6, 8, 5, 1000
if RUN_MODE == "debug":          # tiny synthetic run on CPU, only to prove the code path end-to-end
    CFG.TILE, CFG.SEG_CROP, CFG.MAX_TILES_PER_IMAGE, CFG.MIN_TILES_KEEP = 64, 128, 400, 2
    CFG.MIL_EPOCHS, CFG.SEG_EPOCHS, CFG.SEG_ABL_EPOCHS, CFG.N_FOLDS = 4, 2, 1, 3
    CFG.CROPS_PER_IMAGE, CFG.FEAT_CROPS_PER_CORE, CFG.MC_SAMPLES, CFG.N_BOOT = 3, 3, 3, 100
    CFG.N_CALIB_PER_SCORE, CFG.N_TISSUE_CLUSTERS, CFG.ENC_BATCH, CFG.SEG_BS = 6, 4, 32, 4
    CFG.NUM_WORKERS, CFG.TTA, CFG.RIM_UM = 0, 2, 20.0
    CFG.ENCODER, CFG.SEG_ENCODER, CFG.ENCODER2, CFG.CAM_ARCH = "timm:resnet18", "resnet18", None, "resnet18"
    CFG.N_SEEDS, CFG.CAM_EPOCHS, CFG.CAM_BS = 2, 2, 8

OUT = Path(CFG.OUT); CACHE = OUT / "cache"; FIG = OUT / "figures"; TAB = OUT / "tables"

# ---------------- cross-session resume ----------------
# A Kaggle session wipes /kaggle/working. To continue a stopped run in a NEW session: open the stopped/saved version, then
# "Add Input" -> "Your Work" -> this notebook's output. Any attached folder that contains her2_wsl_v2/cache/calibration.json
# is copied into OUT before anything runs, and every stage below skips the work already present.
CFG.RESUME_FROM = os.environ.get("HER2_RESUME_FROM", None)
def _find_previous_run():
    if CFG.RESUME_FROM: return Path(CFG.RESUME_FROM)
    if not IS_KAGGLE: return None
    for cal in Path("/kaggle/input").rglob("calibration.json"):
        root = cal.parent.parent
        if root.name == OUT.name and (root / "cache" / "feats").exists(): return root
    return None
_prev = _find_previous_run()
_copy_flag = OUT / ".resume_copy_in_progress"
if _prev is not None and _prev.resolve() != OUT.resolve() and (_copy_flag.exists() or not (OUT / "cache" / "calibration.json").exists()):
    import shutil
    OUT.mkdir(parents=True, exist_ok=True); _copy_flag.touch()          # an interrupted copy is simply redone next time
    print(f"RESUME: copying previous run {_prev} -> {OUT} (one-off, a few minutes)")
    shutil.copytree(_prev, OUT, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.zip", ".resume_copy_in_progress"))   # the old review zip is rebuilt at the end
    _copy_flag.unlink(missing_ok=True)
    print("RESUME: done; finished stages will be loaded from cache, unfinished ones continue")

for d in [OUT, CACHE, FIG, TAB, CACHE / "feats", CACHE / "maps", CACHE / "thumbs", CACHE / "mil", CACHE / "logs",
          CACHE / "seg_crops", OUT / "models", OUT / "test_outputs", OUT / "annotation_worklist"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- crash-safe persistence helpers ----------------
# Every cache file is written to a temporary name and atomically renamed, so an interrupted write can never leave a
# truncated file that a restart would mistake for finished work.
def atomic_write(path, write_fn):
    path = Path(path); tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        write_fn(f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def atomic_pickle(obj, path): atomic_write(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))
def atomic_npz(path, compressed=False, **arrays): atomic_write(path, lambda f: (np.savez_compressed if compressed else np.savez)(f, **arrays))
def atomic_torch_save(obj, path): atomic_write(path, lambda f: torch.save(obj, f))
def atomic_csv(df, path, **kw): atomic_write(path, lambda f: f.write(df.to_csv(**kw).encode()))
def load_pickle(path):
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception:
        return None

class ProgressLog:
    """Append-only JSON-lines log for long per-item loops: each finished item is flushed to disk immediately, so a restart
    skips everything already done. A truncated last line (crash mid-write) is ignored and that item is simply redone."""
    def __init__(self, name):
        self.path = CACHE / "logs" / f"{name}.jsonl"; self.items = {}
        if self.path.exists():
            for line in open(self.path):
                try: rec = json.loads(line); self.items[rec["_key"]] = rec
                except Exception: pass
    def __contains__(self, key): return key in self.items
    def __len__(self): return len(self.items)
    def add(self, key, rec):
        rec = dict(rec, _key=key)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"); f.flush(); os.fsync(f.fileno())
        self.items[key] = rec
    def records(self): return list(self.items.values())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else contextlib.nullcontext()

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(CFG.SEED)
torch.backends.cudnn.benchmark = True

WORK_MPP = CFG.NATIVE_MPP * CFG.WORK_DS
THUMB_MPP = WORK_MPP * CFG.THUMB_REL
def um2px(um, mpp=None):
    return um / (mpp or WORK_MPP)

_T0 = {"t": time.time(), "last": time.time()}
SESSION_ID = time.strftime("%Y%m%d-%H%M%S")   # timings from interrupted + resumed sessions are summed per section
# ---------------- memory / disk guard ----------------
# The 10 GB image set lives read-only in /kaggle/input and is streamed one core at a time: it never occupies RAM or the
# /kaggle/working quota. What does consume resources is logged per section (RAM, peak GPU memory, output-folder size).
try: import psutil
except ImportError: psutil = None
def free_disk_gb(path=None):
    import shutil as _sh; return _sh.disk_usage(str(path or OUT)).free / 1e9
def dir_size_gb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: tot += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return tot / 1e9
def resource_report(tag="startup"):
    ram = psutil.virtual_memory() if psutil else None
    gpu = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"[resources:{tag}] RAM total {ram.total / 1e9:.1f} GB, available {ram.available / 1e9:.1f} GB | " if ram else "[resources] ",
          f"GPU {gpu:.1f} GB | free disk in {OUT.parent}: {free_disk_gb():.1f} GB")
    if free_disk_gb() < 12: print("!! WARNING: less than 12 GB free disk; a full run writes about 6-8 GB. Full-resolution test masks "
                                  "will be skipped automatically if space runs low.")
    if ram and ram.total < 12e9: print("!! WARNING: less than 12 GB RAM; set CFG.NUM_WORKERS = 2 if the kernel restarts.")

def section_time(name):
    """Log wall-clock time, current RAM, peak GPU memory and output size per notebook section (computational-cost table),
    then release cached memory before the next section."""
    now = time.time(); dt = now - _T0["last"]; _T0["last"] = now
    gc.collect()
    ram = (psutil.Process().memory_info().rss / 1e9) if psutil else float("nan")
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    disk = dir_size_gb(OUT)
    print(f"[time] {name}: {dt / 60:.1f} min (total {(now - _T0['t']) / 60:.1f} min) | RAM {ram:.1f} GB | "
          f"peak GPU {vram:.1f} GB | output {disk:.1f} GB, free disk {free_disk_gb():.1f} GB")
    import csv
    new = not (OUT / "timings.csv").exists()
    with open(OUT / "timings.csv", "a", newline="") as f:      # csv-quoted: section names contain commas
        w = csv.writer(f)
        if new: w.writerow(["section", "seconds", "session", "ram_gb", "peak_gpu_gb", "output_gb"])
        w.writerow([name, f"{dt:.1f}", SESSION_ID, f"{ram:.2f}", f"{vram:.2f}", f"{disk:.2f}"])

def gpu_safe(fn, x, min_bs=1):
    """Run fn on a batch; on CUDA out-of-memory, free the cache and retry in halves (results are concatenated in order).
    Turns a would-be crash into a slower but identical computation."""
    try: return fn(x)
    except torch.cuda.OutOfMemoryError:
        if len(x) <= min_bs: raise
        torch.cuda.empty_cache(); h = len(x) // 2
        print(f"  (GPU memory full: retrying a batch of {len(x)} as 2 x {h})")
        a, b = gpu_safe(fn, x[:h], min_bs), gpu_safe(fn, x[h:], min_bs)
        return tuple(torch.cat([u, v]) for u, v in zip(a, b)) if isinstance(a, tuple) else torch.cat([a, b])

def save_json(obj, path):
    txt = json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    atomic_write(path, lambda f: f.write(txt.encode()))

# ---------- publication figure style (validated categorical order; text in neutral ink) ----------
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_ORDER = ["0", "1", "2-", "2+", "3"]
GROUP_COL = {"0": "#2a78d6", "1": "#1baf7a", "2-": "#eda100", "2+": "#eb6834", "3": "#e34948"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "lines.linewidth": 2})
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

print(f"RUN_MODE={RUN_MODE} | device={DEVICE} | torch={torch.__version__} | timm={timm.__version__}")
resource_report("startup")
print(f"work level {WORK_MPP:.2f} µm/px | tile {CFG.TILE}px = {CFG.TILE*WORK_MPP:.0f} µm | out={OUT}")
