# ===================== CELL B. ADEL full-resolution regions: stream all archives, keep the most tissue-rich regions per slide =====================
# The Hugging Face archives hold 10,000×10,000 px JPEG tiles at full scan resolution (folders data/HER2_xxx/), with NO µm/px metadata.
# Nuclei look the same size as in our 40x cores, so the native scale is taken as ≈0.25 µm/px (an assumption; section 21 is rerun at
# 0.22 and 0.28 as a sensitivity analysis).
# ADEL slides are mostly needle-core biopsies: thin strips of tissue, so most 2.5 mm tiles are largely glass. v2 therefore does NOT
# require whole tiles to be tissue. It scores every 0.5 mm window of every tile on a 1/8-scale preview and keeps, per slide, the
# N_REG windows with the most tissue (non-overlapping; label-blind). Only tiles that contribute a window are decoded at half resolution.
# This cell streams every archive (≈370 GB) WITHOUT storing it. Restartable: finished archives are recorded and skipped; if the
# selection settings change, old regions and old section-21 logs are cleared automatically. Needs Internet ON and ≥5 GB free disk.
import os, io, time, json, shutil, tarfile, requests, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

REPO = "https://huggingface.co/datasets/aidosSarsembayev/adel_dataset_1/resolve/main"
DST = Path(globals().get("OUT", "/kaggle/working/her2_wsl_v2")) / "ext_adel_regions"; DST.mkdir(parents=True, exist_ok=True)   # inside OUT, so a resumed session copies it back
DONE = DST / "_archives_done.json"
N_REG = int(os.environ.get("HER2_ADEL_REGIONS", 8))        # regions kept per slide
WIN_HALF = 1024                                             # window side at half resolution (= 2048 native px ≈ 0.51 mm at 0.25 µm/px)
MIN_WIN_TISSUE = 0.40                                       # a window needs ≥40% tissue to be eligible
N_ARCHIVES = int(os.environ.get("HER2_ADEL_N_ARCHIVES", 0))  # 0 = all; e.g. 2 for a quick test
ADEL_NATIVE_MPP = float(os.environ.get("HER2_ADEL_NATIVE_MPP", 0.25))
SETTINGS = dict(version=2, N_REG=N_REG, WIN_HALF=WIN_HALF, MIN_WIN_TISSUE=MIN_WIN_TISSUE)

# ---- labels (from the Zenodo zip downloaded in CELL A) ----
lab_p = next(Path("/kaggle/working/ext_adel").rglob("labels.csv"), None)
assert lab_p is not None, "labels.csv not found - run CELL A first."
LAB = pd.read_csv(lab_p); LAB["slide"] = LAB.fname.str.replace(r"\.[a-z]+$", "", regex=True)
print(f"labels: {len(LAB)} slides | score counts {LAB.HER2score.value_counts().sort_index().to_dict()}")

# ---- settings check: regions made with other settings are discarded, together with section-21 results computed on them ----
state = json.load(open(DONE)) if DONE.exists() else None
if not isinstance(state, dict) or state.get("settings") != SETTINGS:
    if state is not None or any(DST.glob("HER2_*")):
        print("Region settings changed (or old v1 regions found): clearing old regions and old ADEL section-21 logs")
        for p in DST.glob("HER2_*"): shutil.rmtree(p, ignore_errors=True)
        for p in DST.glob("_tmp_*"): shutil.rmtree(p, ignore_errors=True)
        if "CACHE" in globals():
            for p in (CACHE / "logs").glob("external_adel*"): p.unlink()
    state = dict(settings=SETTINGS, done=[])
    json.dump(state, open(DONE, "w"))
done = set(state["done"])

# ---- archive list ----
man = requests.get(f"{REPO}/archive_manifest.json", timeout=60).json()
ARCH = sorted(set(k for d in man for k in d) | {"HER2_001_009.tar.gz"})   # the manifest omits the first archive
if N_ARCHIVES: ARCH = ARCH[:N_ARCHIVES]
print(f"{len(ARCH)} archives | already done: {len(done & set(ARCH))}")

def tissue_mask(rgb):
    g = rgb.mean(-1); return (g < 215) & (g > 30)

def candidate_windows(thumb, wt):
    """Non-overlapping wt×wt windows on the 1/8 preview, best first, with their tissue fraction (integral image, stride wt/2)."""
    m = tissue_mask(thumb).astype(np.float32); H, W = m.shape
    if H < wt or W < wt: return []
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1); st = max(1, wt // 2)
    c = [((ii[y + wt, x + wt] - ii[y, x + wt] - ii[y + wt, x] + ii[y, x]) / (wt * wt), y, x)
         for y in range(0, H - wt + 1, st) for x in range(0, W - wt + 1, st)]
    c.sort(reverse=True); picked = []
    for f, y, x in c:
        if f < MIN_WIN_TISSUE or len(picked) >= N_REG: break
        if all(abs(y - py) >= wt or abs(x - px) >= wt for _, py, px in picked): picked.append((f, y, x))
    return picked

def process_archive(name):
    tmp = DST / f"_tmp_{name}"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    best, seen, t0, nbytes, ndec = {}, {}, time.time(), 0, 0      # best[slide] = list of (tissue_frac, file)
    with requests.get(f"{REPO}/data/{name}", stream=True, timeout=120) as r:
        if r.status_code == 404: print(f"  {name}: not on the Hub, skipped"); shutil.rmtree(tmp, ignore_errors=True); return
        r.raise_for_status(); r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
            for m in tf:
                if not m.isfile() or Path(m.name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"): continue
                slide = Path(m.name).parent.name; seen[slide] = seen.get(slide, 0) + 1; nbytes += m.size
                b = tf.extractfile(m).read()
                im = Image.open(io.BytesIO(b)); W0, H0 = im.size; im.draft("RGB", (W0 // 8, H0 // 8)); th = np.asarray(im.convert("RGB"))
                sc = (W0 / 2) / th.shape[1]                                                  # preview px -> half-res px
                wt = max(8, int(round(WIN_HALF / sc)))
                cur = best.setdefault(slide, [])
                cands = [c for c in candidate_windows(th, wt) if len(cur) < N_REG or c[0] > min(cur)[0]]
                if not cands: continue
                im = Image.open(io.BytesIO(b)); im.draft("RGB", (W0 // 2, H0 // 2)); im = im.convert("RGB"); ndec += 1
                if im.size[0] > W0 // 2 + 2: im = im.resize((W0 // 2, H0 // 2), Image.BILINEAR)   # draft not honoured (non-JPEG)
                half = np.asarray(im); d = tmp / slide; d.mkdir(exist_ok=True)
                for f, y, x in cands:
                    if len(cur) >= N_REG and f <= min(cur)[0]: continue
                    Y, X = int(round(y * sc)), int(round(x * sc)); win = half[Y:Y + WIN_HALF, X:X + WIN_HALF]
                    if win.shape[0] < WIN_HALF * 0.9 or win.shape[1] < WIN_HALF * 0.9: continue
                    fp = d / f"{Path(m.name).stem}_y{Y}_x{X}.jpg"; Image.fromarray(win).save(fp, quality=92)
                    cur.append((float(f), str(fp)))
                    if len(cur) > N_REG:
                        cur.sort(); _, worst = cur.pop(0); Path(worst).unlink(missing_ok=True)
    for s in seen:                                                                          # move finished slides into place
        if (tmp / s).exists():
            shutil.rmtree(DST / s, ignore_errors=True); shutil.move(str(tmp / s), str(DST / s))
            json.dump([dict(tissue=f, file=Path(p).name) for f, p in sorted(best.get(s, []), reverse=True)], open(DST / s / "_regions.json", "w"))
    shutil.rmtree(tmp, ignore_errors=True)
    el = time.time() - t0; nk = {s: len(best.get(s, [])) for s in seen}
    mt = [np.mean([f for f, _ in best[s]]) for s in seen if best.get(s)]
    print(f"  {name}: {nbytes/1e9:.1f} GB in {el/60:.1f} min | slides {len(seen)} | tiles {sum(seen.values())} (decoded {ndec}) "
          f"| regions kept {sum(nk.values())} | mean tissue in kept windows {np.mean(mt) if mt else float('nan'):.2f} "
          f"| slides with <{N_REG} regions: { {s: k for s, k in nk.items() if k < N_REG} }", flush=True)

for a in ARCH:
    if a in done: continue
    if shutil.disk_usage("/kaggle/working").free / 1e9 < 3: print("!! less than 3 GB free disk - stopping; free space and rerun"); break
    for attempt in range(3):
        try: process_archive(a); break
        except Exception as e: print(f"  {a}: attempt {attempt + 1} failed ({type(e).__name__}: {e})"); time.sleep(10)
    else: print(f"  {a}: giving up for now; rerun the cell later"); continue
    done.add(a); json.dump(dict(settings=SETTINGS, done=sorted(done)), open(DONE, "w"))

# ---- manifest for section 21 (generic CSV mode): one row per slide folder, scale = 2 × native µm/px (regions are at half resolution) ----
def write_manifest(native_mpp, out=None):
    rows = [dict(path=str(DST / s), ihc_score=int(sc), mpp=2 * native_mpp, n_regions=len(list((DST / s).glob("*.jpg"))))
            for s, sc in zip(LAB.slide, LAB.HER2score) if (DST / s).is_dir() and any((DST / s).glob("*.jpg"))]
    M = pd.DataFrame(rows); out = out or DST / f"manifest_native{native_mpp:.2f}.csv"; M.to_csv(out, index=False); return M, out
M, MP = write_manifest(ADEL_NATIVE_MPP)
miss = sorted(set(LAB.slide) - set(Path(p).name for p in M.path))
if N_ARCHIVES: print(f"(test mode: only {N_ARCHIVES} archives processed, so most labelled slides have no regions yet - expected)")
print(f"\nmanifest {MP}: {len(M)} of {len(LAB)} labelled slides have regions | score counts {M.ihc_score.value_counts().sort_index().to_dict()}")
print(f"regions per slide: {M.n_regions.value_counts().sort_index().to_dict()} | labelled slides without regions: {len(miss)} {miss[:15]}")
print(f"disk used by regions: {sum(f.stat().st_size for f in DST.rglob('*.jpg'))/1e9:.2f} GB")
for alt in (0.22, 0.28): write_manifest(alt)                 # sensitivity manifests (same images, different assumed scale)

# ---- visual check for the supplement: ADEL regions and our cores, both at the 0.46 µm/px work level ----
try:
    import matplotlib.pyplot as plt
    C = 448; fig, ax = plt.subplots(2, 4, figsize=(12, 6.4)); rng = np.random.default_rng(0)
    ex = [list((Path(p)).glob("*.jpg"))[0] for p in M.sample(min(4, len(M)), random_state=0).path]
    for j, p in enumerate(ex):
        im = Image.open(p); f = 2 * ADEL_NATIVE_MPP / 0.46; im = im.resize((int(im.size[0] * f), int(im.size[1] * f)), Image.BILINEAR)
        a = np.asarray(im); y, x = (a.shape[0] - C) // 2, (a.shape[1] - C) // 2; ax[0, j].imshow(a[y:y + C, x:x + C]); ax[0, j].set_title(f"ADEL {p.parent.name}", fontsize=8)
    if "TRAIN" in globals():
        for j, p in enumerate(TRAIN.sample(4, random_state=0).path):
            im = Image.open(p); im = im.resize((im.size[0] // 2, im.size[1] // 2), Image.BILINEAR); a = np.asarray(im.convert("RGB"))
            y, x = (a.shape[0] - C) // 2, (a.shape[1] - C) // 2; ax[1, j].imshow(a[y:y + C, x:x + C]); ax[1, j].set_title("ours (gastric TMA)", fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([]); a_.plot([10, 10 + 50 / 0.46], [C - 14, C - 14], color="k", lw=3); a_.text(10, C - 22, "50 µm", fontsize=7)
    fig.suptitle(f"Both rows at 0.46 µm/px (ADEL assumed native {ADEL_NATIVE_MPP} µm/px): cells should look the same size", fontsize=9)
    plt.tight_layout(); fig.savefig(DST / "adel_vs_ours_work_level.png", dpi=120); plt.show()
except Exception as e: print("figure skipped:", e)

print(f"""
NEXT - run section 21 three times (main analysis + 2 sensitivity runs), each in its own cell:
    os.environ["HER2_EXT_ROOT"] = "{DST}"
    os.environ["HER2_EXT_CSV"]  = "{DST}/manifest_native0.25.csv"; os.environ["HER2_EXT_NAME"] = "adel_mpp025"
    (then paste section 21)
  and the same with manifest_native0.22.csv / adel_mpp022 and manifest_native0.28.csv / adel_mpp028.""")
