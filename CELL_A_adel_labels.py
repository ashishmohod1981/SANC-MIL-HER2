# ===================== CELL A. ADEL labels (Zenodo zip, 0.5 GB) - needed by cell B =====================
import os, zipfile, urllib.request, hashlib
dst = "/kaggle/working/ext_adel"; os.makedirs(dst, exist_ok=True)
if not any(f == "labels.csv" for _, _, fs in os.walk(dst) for f in fs):
    zp = f"{dst}/her2_dataset.zip"
    urllib.request.urlretrieve("https://zenodo.org/records/15872690/files/her2_dataset.zip?download=1", zp)
    md5 = hashlib.md5(open(zp, "rb").read()).hexdigest()
    assert md5 == "35456e147610932fdd829119ef884b34", f"download corrupted (md5 {md5}) - rerun this cell"
    zipfile.ZipFile(zp).extractall(dst); os.remove(zp)
print("labels:", [os.path.join(r, f) for r, _, fs in os.walk(dst) for f in fs if f == "labels.csv"])
