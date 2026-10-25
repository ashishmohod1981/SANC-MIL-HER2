# Derived data

All files below were produced by the pipeline in this repository. Source images are not redistributed; download them from the original datasets (see README).

## GEA cohort (`results/gea/`)

| File | Contents |
|---|---|
| `tables/classification_table.csv`, `classification_full.csv` | Out-of-fold and test metrics (AUROC, AUPRC, balanced accuracy, sensitivity, specificity, MCC, ECE) with 95 % bootstrap CIs for every method |
| `tables/paired_delta_auc.csv` | Paired bootstrap AUROC differences, SANC-MIL minus each method, with Holm-corrected p-values |
| `tables/localization_proxies.csv` | Label-grounded localisation metrics (IHC 0 false-positive fractions, 3+ coverage, tile AUROC) and proxies (fold stability, stain concordance, deletion test) |
| `tables/seed_variability.csv`, `seed_runs.csv` | Results over three seeds |
| `tables/h1_negative_constraint_tests.csv` | Per-seed paired tests of the negative-bag constraint on test and out-of-fold cores |
| `tables/h3_feature_separation_3v2_and_fish.csv` | Single-feature AUROCs for 3+ vs 2+ and FISH status within 2+ |
| `tables/clinical_feature_models.csv`, `clinical_feature_trends.csv` | Membrane-feature models and trends with IHC score |
| `tables/test_predictions.csv` | Per-core test predictions: probability, entropy, abstention, ordinal probabilities, ISH routing and final decision |
| `tables/test_error_by_ihc_group.csv`, `abstention.json`, `ordinal_score_metrics.json` | Decision analysis |
| `tables/folds.csv`, `proxy_groups.json` | Proxy-patient group and fold of every training core |
| `tables/qc_*.csv`, `qc_summary.json`, `leakage_sensitivity.csv`, `near_duplicates.csv`, `data_validation.json` | Data checks, quality control and leakage sensitivity |
| `derived/calibration.json` | Macenko stain matrix, agreement with Ruifrok vectors and label-anchored DAB thresholds (OD) |
| `derived/core_features.csv` | Core-level membrane and cell-coverage features (student and rule-based) for every core |
| `derived/seg_*_metrics.json`, `seg_test_proxies.csv`, `test_seg_summary.csv` | Segmentation-student agreement with pseudo-labels and test-set segmentation summary |
| `environment_and_config.json`, `timings.csv` | Package versions, full configuration and run times |

## ADEL cohort

| Path | Contents |
|---|---|
| `results/adel_external/mpp0xx/*_per_item.csv` (xx = 22, 25, 28) | Per-slide predictions of every model at the assumed pixel size, frozen and re-estimated stain matrix |
| `results/adel_external/*/*_summary.csv`, `*_h1_negative_constraint.csv` | Summary metrics and paired tests of the negative-bag constraint |
| `results/adel_site_adaptation/*.csv` | Site adaptation: per-slide out-of-fold predictions, per-repeat results, thresholds and summary with the two primary end points |
| `results/derived/recovered.txt` | The 48 slides re-processed with the optical-density tissue test |

Licence of the derived data: CC BY 4.0. Please also cite the original datasets.
