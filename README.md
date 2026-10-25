# SANC-MIL: exact negative bags for HER2 localisation

Code and results for the manuscript

> **Exact negative bags from routine HER2 scores reduce and measure false-positive localisation in weakly supervised immunohistochemistry models**
> Ashish P. Mohod, K. P. Yadav, Bhavana Mohod. Submitted to *Medical & Biological Engineering & Computing*.

A core that a pathologist scored HER2 IHC 0 or 1+ contains no HER2-positive tissue, so every tile in it is a true negative. SANC-MIL (stain-anchored, negative-constrained multiple-instance learning) uses that fact twice:

* **during training**: a negative-bag loss pushes every tile logit of IHC 0/1+ cores towards negative;
* **during evaluation**: the fraction of tiles called positive inside IHC 0 cores is a false-positive rate with a known denominator.

The repository contains the code and all **derived data** of the study, but **no source images**. Both datasets are public and must be downloaded from their original sources (see *Data*).

## Repository layout

| Path | Contents |
|---|---|
| `src/p01_setup.py` … `src/p11_external.py` | Pipeline modules, in run order: setup and configuration, data checks, stain calibration and tile features, QC and proxy-patient groups, MIL models and comparators, localisation metrics, pseudo-labels and segmentation student, cell features, validation report, export, external evaluation |
| `build.py` | Concatenates `src/` into one script and one Kaggle notebook |
| `notebooks/her2_wsl_v2_kaggle.ipynb` | The notebook used for the reported GEA run (Kaggle, one Tesla T4, seed 2024) |
| `external_adel/CELL_A…E2_*.py` | Cells appended to the notebook for the ADEL breast cohort (run in the order below) |
| `figures/scripts/` | Scripts that draw every manuscript figure from the result tables |
| `results/gea/` | GEA result tables, run configuration and package versions, fold assignment |
| `results/adel_external/mpp0xx/` | Per-slide predictions and summaries at the three assumed pixel sizes |
| `results/adel_site_adaptation/` | Per-slide, per-repeat and summary results of the site-adaptation analysis |
| `results/derived/` | Values computed from the per-slide files for Table 5 and Figs. 8–9 |
| `results/gea/derived/` | Stain calibration (matrix and DAB thresholds), core-level membrane and cell features, segmentation-student metrics, test-set segmentation summary |
| `results/gea/figures/`, `results/gea/test_overlays/` | Diagnostic figures and whole-core membrane overlays written by the pipeline |
| `results/gea/annotation_worklist/` | Protocol, regions and QuPath scripts prepared for a future pathologist annotation study |
| `DATA.md` | Description of every data file and its columns |

## Model at a glance

* Tiles: 224 × 224 px at 0.46 µm/px with ≥ 50 % tissue, up to 400 per core.
* Fixed descriptors: frozen ImageNet ResNet-50 truncated after stage 3 (1,024-d) and 14 calibrated stain features (Macenko matrix, label-anchored DAB thresholds, membrane-shape test).
* Trained head (346,181 parameters): stain projection 14→64; fusion 1,088→256 with dropout 0.25; gated attention 256→128→1; instance head 256→1; additive bag logit `z = Σ a_t s_t`; CORAL ordinal head on `u = Σ a_t x_t`.
* Loss: `L = L_status + 0.5 L_ord + 0.5 L_neg + 0.5 L_top-k`, where `L_neg` acts on every tile of IHC 0/1+ bags and `L_top-k` on the top 5 % of tiles of HER2-positive bags.
* Training: AdamW (2e-4, weight decay 1e-3), one-cycle schedule, 40 epochs, 16 bags per batch, ≤ 256 tiles per bag, 10 % tile dropout, gradient clipping 1.0; 5 proxy-patient grouped folds; 3 seeds for key models.

## Data

* **GEA** (gastro-oesophageal adenocarcinoma, HER2 IHC TMA cores; Pisula et al.): https://doi.org/10.5281/zenodo.7031868. Attach the full-resolution cores as a Kaggle dataset input (named `her2dataset` in the notebook).
* **ADEL** (breast HER2 IHC slides; Dunenova et al.): labels https://doi.org/10.5281/zenodo.15872690; full-resolution tiles https://huggingface.co/datasets/aidosSarsembayev/adel_dataset_1. Cell A downloads the labels and cell B streams the tiles.

Both datasets are released under CC BY 4.0 by their authors. Please cite the original publications when you use them.

## How to reproduce

### 1. GEA pipeline (about 6–7 h on a T4)
1. Create a new Kaggle notebook and import `notebooks/her2_wsl_v2_kaggle.ipynb`.
2. Add the GEA images as input, set the accelerator to **GPU T4** and turn **Internet on** (ImageNet and Phikon weights are downloaded).
3. *Save & Run All*. Results are written to `/kaggle/working/her2_wsl_v2/`. The run is restartable: every finished step is written atomically and skipped on re-run.

### 2. ADEL external evaluation and site adaptation
Append the cells in `external_adel/` to the same notebook session (they reuse the trained fold models) and run them in this order:

| Cell | Purpose | Approx. time |
|---|---|---|
| A | Download ADEL labels | minutes |
| B | Stream full-resolution tiles; keep up to 8 tissue-rich 0.5 mm windows per slide | 1–2 h |
| B2 | Re-process the 48 slides with no qualifying window (optical-density tissue test) | 20–40 min |
| C | External analysis at 0.25 µm/px (frozen and re-estimated stain) | 20–30 min |
| D | Sensitivity runs at 0.22 and 0.28 µm/px; bundle outputs | 40–60 min |
| E1 | Extended analysis from the per-slide files (no model re-run) | 1–2 min |
| E2 | Site adaptation (5-fold CV × 3 repeats; protocol fixed in the cell header) | 40–70 min |

### 3. Figures
```bash
pip install -r requirements.txt
python figures/scripts/data_figs.py      # Figs. 4, 5, 7 and ESM Figs. S3, S4, S7
python figures/scripts/ext_fig.py        # Fig. 8 (external evaluation)
python figures/scripts/fig11.py          # Fig. 9 (site adaptation)
python figures/scripts/maps_fig.py       # Fig. 6 (example maps)
python figures/scripts/make_diagrams_v2.py   # Figs. 1–3 and ESM Fig. S6
```
Outputs go to `figures/output/`. Fig. 2a needs an example IHC core image at `figures/assets/example_core.jpg`, which is not distributed here; any GEA core can be used. Fonts: Liberation Sans (metric-compatible with Arial).

## Main results (from the files in `results/`)

| Measure | SANC-MIL | Without negative-bag loss |
|---|---|---|
| GEA test AUROC (307 cores, 3 seeds) | 0.993 ± 0.000 | 0.992 ± 0.002 |
| GEA IHC 0 false-positive tile fraction (284 cores) | 0.005 ± 0.001 | 0.082 ± 0.015 |
| ADEL IHC 0 false-positive fraction, frozen models (138 slides) | 0.053 | 0.185 |
| ADEL status AUROC, status logit / ordinal head | 0.856 / 0.979 | 0.933 / 0.977 |
| ADEL after site adaptation: status AUROC; IHC 0 FP fraction | 0.940; 0.015 | 0.949; 0.086 |

## Software
Recorded versions of the reported run: PyTorch 2.11.0 (CUDA 12.8), timm 1.0.29, NumPy 2.1.3, scikit-learn 1.6.1, scikit-image 0.25.2, OpenCV 4.14.0 (`results/gea/environment_and_config.json`).

## Citation
If you use this code, please cite the article (details will be added after publication) and the two datasets. A `CITATION.cff` file is included.

## Licence
Code: MIT licence (see `LICENSE`). Result tables: CC BY 4.0. The datasets keep their own licences.

## Contact
Ashish P. Mohod, mohod.ashish@gmail.com (ORCID 0009-0008-0170-5199)
