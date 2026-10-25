# ===================== CELL B2. Recover ADEL slides that cell B could not use =====================
# Cell B's tissue test (grey value < 215) misses very pale tissue, which is mostly IHC 0/1+ (33 of 39 failures), so leaving
# those slides out would bias the external cohort towards stained slides. This cell revisits ONLY the missing slides:
#   * slides whose windows all failed the tissue test -> re-selected with an optical-density tissue test that detects pale
#     tissue (OD ≥ 0.06 above the tile's own background), window threshold 25% tissue;
#   * the unfinished archive (HER2_130_139) -> processed with the same relaxed rule, with longer time-outs and 4 retries.
# Selection stays label-blind (tissue content only). Recovered slides are listed in recovered_slides.csv so the paper can
# report results with and without them. Run after cell B, before cells C and D. Needs Internet ON.
import os, io, time, json, shutil, tarfile, requests, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

REPO = "https://huggingface.co/datasets/aidosSarsembayev/adel_dataset_1/resolve/main"
DST = OUT / "ext_adel_regions"
N_REG, WIN_HALF, MIN_WIN_TISSUE_RELAXED, OD_TISSUE = 8, 1024, 0.25, 0.06
ADEL_NATIVE_MPP = 0.25

lab = pd.read_csv(next(Path('/kaggle/working/ext_adel').rglob('labels.csv')))
lab["slide"] = lab.fname.str.replace(r"\.[a-z]+$", "", regex=True)
man = requests.get(f"{REPO}/archive_manifest.json", timeout=60).json()
arch_of = {s: a for d in man for a, ss in d.items() for s in ss}
for i in range(1, 10): arch_of.setdefault(f"HER2_{i:03d}", "HER2_001_009.tar.gz")
has = lambda s: (DST / s).is_dir() and any((DST / s).glob("*.jpg"))
missing = [s for s in lab.slide if not has(s) and s in arch_of]
todo = {}
for s in missing: todo.setdefault(arch_of[s], set()).add(s)
print(f"slides to recover: {len(missing)} in {len(todo)} archives")

def tissue_mask_od(rgb):
    """Pale-tissue-aware mask: optical density above the tile's own bright background."""
    x = rgb.astype(np.float32); I0 = np.percentile(x.reshape(-1, 3), 99.5, axis=0).clip(180, 255)
    od = (-np.log((x + 1) / (I0 + 1))).clip(0, None).max(-1)
    return (od >= OD_TISSUE) & (x.mean(-1) > 30)

def windows_relaxed(thumb, wt):
    m = tissue_mask_od(thumb).astype(np.float32); H, W = m.shape
    if H < wt or W < wt: return []
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1); st = max(1, wt // 2)
    c = sorted(((ii[y + wt, x + wt] - ii[y, x + wt] - ii[y + wt, x] + ii[y, x]) / (wt * wt), y, x)
               for y in range(0, H - wt + 1, st) for x in range(0, W - wt + 1, st))[::-1]
    picked = []
    for f, y, x in c:
        if f < MIN_WIN_TISSUE_RELAXED or len(picked) >= N_REG: break
        if all(abs(y - py) >= wt or abs(x - px) >= wt for _, py, px in picked): picked.append((f, y, x))
    return picked

def recover_archive(name, slides):
    tmp = DST / f"_tmp_recover_{name}"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    best, t0, skipped = {s: [] for s in slides}, time.time(), []
    with requests.get(f"{REPO}/data/{name}", stream=True, timeout=300) as r:
        r.raise_for_status(); r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
            for m in tf:
                if not m.isfile() or Path(m.name).suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"): continue
                slide = Path(m.name).parent.name
                if slide not in slides: continue                                  # other slides are already done
                b = tf.extractfile(m).read()
                try:
                    im = Image.open(io.BytesIO(b)); W0, H0 = im.size
                    if min(W0, H0) < 2 * WIN_HALF:                                # tiny/odd file (thumbnail, label image):
                        skipped.append((m.name, W0, H0)); continue                # PIL's draft() divides by W0//8 -> crash
                    im.draft("RGB", (W0 // 8, H0 // 8)); th = np.asarray(im.convert("RGB"))
                    sc = (W0 / 2) / th.shape[1]; wt = max(8, int(round(WIN_HALF / sc))); cur = best[slide]
                    cands = [c for c in windows_relaxed(th, wt) if len(cur) < N_REG or c[0] > min(cur)[0]]
                    if not cands: continue
                    im = Image.open(io.BytesIO(b)); im.draft("RGB", (W0 // 2, H0 // 2)); im = im.convert("RGB")
                    if im.size[0] > W0 // 2 + 2: im = im.resize((W0 // 2, H0 // 2), Image.BILINEAR)
                    half = np.asarray(im); d = tmp / slide; d.mkdir(exist_ok=True)
                    for f, y, x in cands:
                        if len(cur) >= N_REG and f <= min(cur)[0]: continue
                        Y, X = int(round(y * sc)), int(round(x * sc)); win = half[Y:Y + WIN_HALF, X:X + WIN_HALF]
                        if win.shape[0] < WIN_HALF * 0.9 or win.shape[1] < WIN_HALF * 0.9: continue
                        fp = d / f"{Path(m.name).stem}_y{Y}_x{X}.jpg"; Image.fromarray(win).save(fp, quality=92); cur.append((float(f), str(fp)))
                        if len(cur) > N_REG: cur.sort(); _, worst = cur.pop(0); Path(worst).unlink(missing_ok=True)
                except Exception as e:                                           # one unreadable tile must not kill the archive
                    skipped.append((m.name, type(e).__name__, str(e)[:80]))
    got = []
    for s in slides:
        if (tmp / s).exists() and best[s]:
            shutil.rmtree(DST / s, ignore_errors=True); shutil.move(str(tmp / s), str(DST / s))
            json.dump([dict(tissue=f, file=Path(p).name, rule="relaxed_OD") for f, p in sorted(best[s], reverse=True)], open(DST / s / "_regions.json", "w"))
            got.append(s)
    shutil.rmtree(tmp, ignore_errors=True)
    if skipped: print(f"  {name}: skipped {len(skipped)} unusable file(s), e.g. {skipped[:3]}")
    print(f"  {name}: recovered {len(got)}/{len(slides)} slides in {(time.time() - t0) / 60:.1f} min {sorted(set(slides) - set(got)) or ''}", flush=True)
    return got

rec_path = DST / "recovered_slides.csv"
recovered = set(pd.read_csv(rec_path).slide) if rec_path.exists() else set()
for a in sorted(todo):
    for attempt in range(4):
        try: recovered |= set(recover_archive(a, todo[a])); break
        except Exception as e: print(f"  {a}: attempt {attempt + 1} failed ({type(e).__name__}: {str(e)[:150]})"); time.sleep(30)
    pd.DataFrame({"slide": sorted(recovered)}).to_csv(rec_path, index=False)

# rewrite the three manifests (same images for every assumed scale)
for mpp in (0.22, 0.25, 0.28):
    rows = [dict(path=str(DST / s), ihc_score=int(sc), mpp=2 * mpp, n_regions=len(list((DST / s).glob("*.jpg"))))
            for s, sc in zip(lab.slide, lab.HER2score) if has(s)]
    pd.DataFrame(rows).to_csv(DST / f"manifest_native{mpp:.2f}.csv", index=False)
lab["have"] = lab.slide.map(has)
print(f"\nrecovered: {len(recovered)} slides | now usable: {lab.have.sum()} of {len(lab)}")
print(pd.crosstab(lab.HER2score, lab.have).rename(columns={True: "usable", False: "still missing"}))
print("still missing:", list(lab[~lab.have].slide))
