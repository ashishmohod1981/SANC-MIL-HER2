# %% [markdown]
# ## 20. Export a review package
# This cell zips the tables, figures, summaries, timings and environment, plus a stratified sample of masks and pseudo-labels. Download
# `her2_v2_results_for_review.zip` from the Output panel.

# %%
import glob, zipfile
Z = Path("/kaggle/working/her2_v2_results_for_review.zip" if IS_KAGGLE else OUT.parent / "her2_v2_results_for_review.zip")
files = glob.glob(f"{OUT}/tables/*") + glob.glob(f"{OUT}/figures/*.png") + glob.glob(f"{OUT}/*.md") + glob.glob(f"{OUT}/*.json") + glob.glob(f"{OUT}/*.csv")
files += glob.glob(f"{CACHE}/*.json") + glob.glob(f"{CACHE}/*.csv")
files += glob.glob(f"{OUT}/annotation_worklist/*.csv") + glob.glob(f"{OUT}/annotation_worklist/*.md") + glob.glob(f"{OUT}/annotation_worklist/*.groovy")
files += glob.glob(f"{OUT}/annotation_worklist/qupath_geojson/*.geojson")
p_ = pd.read_csv(TAB / "test_predictions.csv"); ids = list(p_[p_.group.isin(["2-", "2+", "3"])].image_id)
s_ = pd.read_csv(CACHE / "test_seg_summary.csv"); z_ = s_[s_.group == "0"]
ids += list(z_.sort_values("stu_mem_mod", ascending=False).image_id[:12]) + list(z_.sample(min(6, len(z_)), random_state=0).image_id)
files += [f for i in ids for f in glob.glob(f"{OUT}/test_outputs/{i}_overlay.jpg")]
c_ = pd.read_csv(CACHE / "seg_crops_index.csv").groupby("group", group_keys=False).apply(lambda d: d.sample(min(4, len(d)), random_state=0))
files += list(c_.crop) + list(c_.label)
files += sorted(glob.glob(f"{OUT}/annotation_worklist/rois/*.png"))[:6] + sorted(glob.glob(f"{OUT}/annotation_worklist/pred_masks/*.png"))[:6]
with zipfile.ZipFile(Z, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(set(files)):
        if os.path.exists(f): zf.write(f, os.path.relpath(f, OUT.parent))
print(f"{Z}  {Z.stat().st_size / 1e6:.1f} MB  ({len(set(files))} files)")
