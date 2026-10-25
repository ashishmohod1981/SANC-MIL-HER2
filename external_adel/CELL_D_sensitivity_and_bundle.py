# ===================== CELL D. Sensitivity runs (assumed native scale 0.22 and 0.28 µm/px) + one zip to send back =====================
# Uses the functions defined in CELL C; only the manifest (scale) and the output name change. Same images, same models.
import zipfile, glob
for tag, mpp in [("adel_mpp022", "0.22"), ("adel_mpp028", "0.28")]:
    EXT_CSV = str(OUT / "ext_adel_regions" / f"manifest_native{mpp}.csv"); EXT_NAME = tag
    _ext = run_external()
    if _ext is not None:
        analyse_external(*_ext); section_time(f"external validation ({tag})")

bundle = OUT.parent / "external_adel_all_results.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    files = sorted(glob.glob(str(OUT / "external_adel_mpp0*_results.zip"))) + \
            [str(p) for p in (OUT / "ext_adel_regions").glob("*.csv")] + \
            [str(p) for p in (OUT / "ext_adel_regions").glob("*.png")] + [str(TAB / "compute_section_times.csv")]
    for f in files:
        if os.path.exists(f): z.write(f, os.path.basename(f))
print("SEND THIS FILE:", bundle, f"({os.path.getsize(bundle)/1e6:.1f} MB)"); print("contains:", [os.path.basename(f) for f in files])
