# Results summary (auto-generated)

*Run mode:* `full` | encoder `timm:resnet50.tv_in1k` (pretrained=True) | train cores 1554 | test cores 307 (16 HER2+) | test groups: 0: 284, 1: 0, 2-: 7, 2+: 8, 3: 8

## Classification (temporally independent test set)

| method                             |   OOF AUROC (grouped CV) | Test AUROC          | Test AUPRC          | Test BalAcc         | Test Sens           | Test Spec           | Test MCC            |   Test ECE |
|:-----------------------------------|-------------------------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|-----------:|
| Stain intensity rule (Pisula 2023) |                    0.963 | 0.970 [0.940–0.992] | 0.679 [0.489–0.887] | 0.885 [0.808–0.931] | 0.938 [0.812–1.000] | 0.832 [0.787–0.873] | 0.421 [0.334–0.498] |      0.152 |
| Calibrated stain features + LR     |                    0.98  | 0.988 [0.970–0.999] | 0.897 [0.775–0.985] | 0.931 [0.912–0.952] | 1.000 [1.000–1.000] | 0.863 [0.825–0.904] | 0.496 [0.444–0.573] |      0.11  |
| MeanPool                           |                    0.945 | 0.959 [0.896–0.997] | 0.828 [0.666–0.965] | 0.898 [0.803–0.967] | 0.875 [0.688–1.000] | 0.921 [0.890–0.952] | 0.543 [0.416–0.664] |      0.052 |
| MaxPool                            |                    0.969 | 0.978 [0.950–0.997] | 0.827 [0.672–0.955] | 0.905 [0.831–0.950] | 0.938 [0.812–1.000] | 0.873 [0.835–0.911] | 0.480 [0.387–0.566] |      0.043 |
| ABMIL (Ilse; BJC-2023)             |                    0.943 | 0.948 [0.873–0.995] | 0.788 [0.603–0.939] | 0.905 [0.808–0.973] | 0.875 [0.688–1.000] | 0.935 [0.904–0.962] | 0.581 [0.446–0.713] |      0.054 |
| CLAM-SB                            |                    0.947 | 0.941 [0.858–0.994] | 0.772 [0.586–0.930] | 0.903 [0.806–0.971] | 0.875 [0.688–1.000] | 0.931 [0.900–0.959] | 0.571 [0.439–0.700] |      0.051 |
| TransMIL-lite                      |                    0.949 | 0.985 [0.970–0.995] | 0.788 [0.630–0.928] | 0.948 [0.879–0.988] | 0.938 [0.812–1.000] | 0.959 [0.935–0.979] | 0.703 [0.587–0.829] |      0.071 |
| SANC-MIL (proposed)                |                    0.982 | 0.993 [0.984–0.999] | 0.904 [0.796–0.988] | 0.925 [0.830–0.991] | 0.875 [0.688–1.000] | 0.976 [0.959–0.990] | 0.749 [0.601–0.888] |      0.042 |
| SANC − stain features              |                    0.957 | 0.963 [0.911–0.995] | 0.783 [0.606–0.930] | 0.907 [0.811–0.974] | 0.875 [0.688–1.000] | 0.938 [0.911–0.966] | 0.591 [0.453–0.726] |      0.047 |
| SANC − negative constraint         |                    0.975 | 0.993 [0.983–0.999] | 0.909 [0.803–0.989] | 0.948 [0.877–0.988] | 0.938 [0.812–1.000] | 0.959 [0.935–0.979] | 0.703 [0.576–0.824] |      0.056 |
| SANC − top-k constraint            |                    0.984 | 0.993 [0.982–0.999] | 0.908 [0.804–0.986] | 0.943 [0.870–0.985] | 0.938 [0.812–1.000] | 0.948 [0.921–0.973] | 0.663 [0.548–0.788] |      0.047 |
| SANC − ordinal head                |                    0.978 | 0.993 [0.983–0.999] | 0.903 [0.792–0.989] | 0.974 [0.960–0.986] | 1.000 [1.000–1.000] | 0.948 [0.921–0.973] | 0.700 [0.615–0.805] |      0.042 |
| SANC − additive (attn pool)        |                    0.974 | 0.993 [0.983–0.999] | 0.916 [0.816–0.988] | 0.925 [0.830–0.991] | 0.875 [0.688–1.000] | 0.976 [0.959–0.990] | 0.749 [0.601–0.888] |      0.047 |
| Stain-only MIL (no deep emb)       |                    0.98  | 0.995 [0.987–0.999] | 0.921 [0.824–0.990] | 0.953 [0.884–0.991] | 0.938 [0.812–1.000] | 0.969 [0.948–0.986] | 0.751 [0.627–0.877] |      0.045 |
| SANC-MIL (proposed) [phikon]       |                    0.975 | 0.979 [0.953–0.996] | 0.755 [0.573–0.933] | 0.938 [0.865–0.979] | 0.938 [0.812–1.000] | 0.938 [0.911–0.962] | 0.628 [0.515–0.740] |      0.06  |
| SANC − stain features [phikon]     |                    0.972 | 0.977 [0.950–0.996] | 0.725 [0.544–0.930] | 0.926 [0.853–0.969] | 0.938 [0.812–1.000] | 0.914 [0.880–0.945] | 0.562 [0.452–0.664] |      0.064 |
| ABMIL (Ilse; BJC-2023) [phikon]    |                    0.957 | 0.984 [0.969–0.995] | 0.681 [0.522–0.923] | 0.934 [0.863–0.976] | 0.938 [0.812–1.000] | 0.931 [0.900–0.959] | 0.608 [0.498–0.726] |      0.1   |
| MaxPool [phikon]                   |                    0.975 | 0.984 [0.963–0.998] | 0.861 [0.723–0.972] | 0.884 [0.783–0.974] | 0.812 [0.625–1.000] | 0.955 [0.931–0.979] | 0.613 [0.455–0.768] |      0.055 |
| Grad-CAM CNN (thumbnail)           |                    0.979 | 0.998 [0.994–1.000] | 0.968 [0.913–1.000] | 0.973 [0.959–0.985] | 1.000 [1.000–1.000] | 0.945 [0.918–0.969] | 0.687 [0.606–0.788] |      0.037 |

Paired bootstrap ΔAUROC, SANC-MIL (proposed) vs ABMIL (Ilse; BJC-2023): +0.046 (95% CI +0.003 to +0.113, p=0.007).

All paired comparisons with the proposed model (bootstrap p, Holm-corrected):

| comparison                                               |    dAUC |      lo |     hi |   p_boot |   p_holm |
|:---------------------------------------------------------|--------:|--------:|-------:|---------:|---------:|
| SANC-MIL (proposed) − Stain intensity rule (Pisula 2023) |  0.023  |  0.0052 | 0.0462 |    0.002 |    0.036 |
| SANC-MIL (proposed) − Calibrated stain features + LR     |  0.0049 | -0.0024 | 0.017  |    0.322 |    1     |
| SANC-MIL (proposed) − MeanPool                           |  0.0338 |  0.0009 | 0.0874 |    0.027 |    0.243 |
| SANC-MIL (proposed) − MaxPool                            |  0.0148 |  0.0019 | 0.0372 |    0.006 |    0.078 |
| SANC-MIL (proposed) − ABMIL (Ilse; BJC-2023)             |  0.0456 |  0.0026 | 0.1128 |    0.007 |    0.084 |
| SANC-MIL (proposed) − CLAM-SB                            |  0.0528 |  0.0036 | 0.1261 |    0.002 |    0.036 |
| SANC-MIL (proposed) − TransMIL-lite                      |  0.0082 |  0.0015 | 0.0174 |    0.015 |    0.154 |
| SANC-MIL (proposed) − SANC − stain features              |  0.0302 |  0.0024 | 0.0754 |    0.004 |    0.064 |
| SANC-MIL (proposed) − SANC − negative constraint         |  0.0001 | -0.0021 | 0.0026 |    1     |    1     |
| SANC-MIL (proposed) − SANC − top-k constraint            |  0.0005 | -0.0024 | 0.0039 |    0.76  |    1     |
| SANC-MIL (proposed) − SANC − ordinal head                |  0.0004 | -0.0015 | 0.0032 |    0.973 |    1     |
| SANC-MIL (proposed) − SANC − additive (attn pool)        | -0.0002 | -0.0026 | 0.0024 |    0.947 |    1     |
| SANC-MIL (proposed) − Stain-only MIL (no deep emb)       | -0.0015 | -0.0045 | 0      |    0.146 |    1     |
| SANC-MIL (proposed) − SANC-MIL (proposed) [phikon]       |  0.014  |  0.0017 | 0.0339 |    0.005 |    0.07  |
| SANC-MIL (proposed) − SANC − stain features [phikon]     |  0.0157 |  0.0017 | 0.0374 |    0.004 |    0.064 |
| SANC-MIL (proposed) − ABMIL (Ilse; BJC-2023) [phikon]    |  0.0094 |  0.0015 | 0.0195 |    0.014 |    0.154 |
| SANC-MIL (proposed) − MaxPool [phikon]                   |  0.0088 | -0.0002 | 0.0228 |    0.072 |    0.576 |
| SANC-MIL (proposed) − Grad-CAM CNN (thumbnail)           | -0.0046 | -0.0127 | 0.0011 |    0.148 |    1     |

Leakage sensitivity (OOF AUROC, random vs proxy-grouped CV):

| method                 |   OOF_AUROC_random_CV |   OOF_AUROC_proxy_grouped_CV |
|:-----------------------|----------------------:|-----------------------------:|
| SANC-MIL (proposed)    |                0.9691 |                       0.9823 |
| ABMIL (Ilse; BJC-2023) |                0.9546 |                       0.9427 |

Abstention / equivocal routing:

```
{
 "abstain_rate_test": 0.1368078175895765,
 "balacc_all": 0.9254725085910653,
 "balacc_retained": 0.9940239043824701,
 "ISH_routing_sensitivity_for_true_IHC2": 0.9333333333333333,
 "ISH_routing_rate_in_non_IHC2": 0.1643835616438356,
 "ISH_threshold_P_IHC2": 0.2040740802329286,
 "ISH_routing_OOF_sensitivity": 0.7945205479452054,
 "ISH_routing_OOF_rate_non_IHC2": 0.1411208642808913,
 "ISH_routing_test_rate_IHC0": 0.16901408450704225,
 "ISH_routing_test_rate_IHC3": 0.0,
 "AURC_proposed": 0.012847420948980497,
 "AURC_abmil": 0.0507274701087567
}
```

## Localisation (grounded checks + proxies)

| method                                  | score_type   |   nonfinite_tile_scores |   fold_stability_spearman |   top10pct_jaccard |   IHC0_tile_FP_rate@0.5 |   IHC0_tile_FP_rate@0.5_lo |   IHC0_tile_FP_rate@0.5_hi |   OOF_IHC0_tile_FP_rate@0.5 |   OOF_IHC0_cores_with_any_tile>0.9 |   IHC0_tile_FP_rate@0.9 |   IHC0_cores_with_any_tile>0.9 |   IHC3_cores_detected(any tile>0.5) |   IHC3_median_pos_tile_frac |   tile_AUROC_IHC3_vs_IHC0 |   stain_concordance_rho(pos cores) |   deletion_drop@10%_top_minus_random |
|:----------------------------------------|:-------------|------------------------:|--------------------------:|-------------------:|------------------------:|---------------------------:|---------------------------:|----------------------------:|-----------------------------------:|------------------------:|-------------------------------:|------------------------------------:|----------------------------:|--------------------------:|-----------------------------------:|-------------------------------------:|
| MeanPool                                | positivity   |                       0 |                     0.862 |              0.556 |                   0.051 |                      0.041 |                      0.063 |                       0.043 |                              0.051 |                   0.004 |                          0.063 |                                   1 |                       0.885 |                     0.966 |                              0.783 |                              nan     |
| MaxPool                                 | positivity   |                       0 |                     0.833 |              0.501 |                   0.001 |                      0     |                      0.001 |                       0     |                              0.001 |                   0     |                          0     |                                   1 |                       0.653 |                     0.932 |                              0.847 |                              nan     |
| ABMIL (Ilse; BJC-2023)                  | attention    |                       0 |                     0.582 |              0.246 |                 nan     |                    nan     |                    nan     |                     nan     |                            nan     |                 nan     |                        nan     |                                 nan |                     nan     |                   nan     |                            nan     |                                0.054 |
| CLAM-SB                                 | positivity   |                       0 |                     0.554 |              0.291 |                   0.003 |                      0.001 |                      0.004 |                       0.004 |                              0.002 |                   0     |                          0.004 |                                   1 |                       0.715 |                     0.894 |                              0.772 |                              nan     |
| TransMIL-lite                           | none         |                     nan |                   nan     |            nan     |                 nan     |                    nan     |                    nan     |                     nan     |                            nan     |                 nan     |                        nan     |                                 nan |                     nan     |                   nan     |                            nan     |                              nan     |
| SANC-MIL (proposed)                     | positivity   |                       0 |                     0.635 |              0.425 |                   0.005 |                      0.003 |                      0.008 |                       0.005 |                              0.012 |                   0.001 |                          0.028 |                                   1 |                       0.909 |                     0.97  |                              0.92  |                                0.065 |
| SANC − stain features                   | positivity   |                       0 |                     0.838 |              0.611 |                   0.012 |                      0.007 |                      0.019 |                       0.007 |                              0.011 |                   0.001 |                          0.028 |                                   1 |                       0.792 |                     0.944 |                              0.837 |                              nan     |
| SANC − negative constraint              | positivity   |                       0 |                     0.427 |              0.387 |                   0.075 |                      0.057 |                      0.098 |                       0.087 |                              0.353 |                   0.026 |                          0.261 |                                   1 |                       0.954 |                     0.963 |                              0.927 |                              nan     |
| SANC − top-k constraint                 | positivity   |                       0 |                     0.721 |              0.424 |                   0.002 |                      0.001 |                      0.003 |                       0.001 |                              0.003 |                   0.001 |                          0.021 |                                   1 |                       0.877 |                     0.969 |                              0.924 |                              nan     |
| SANC − ordinal head                     | positivity   |                       0 |                     0.579 |              0.4   |                   0.006 |                      0.004 |                      0.009 |                       0.003 |                              0.011 |                   0.001 |                          0.028 |                                   1 |                       0.906 |                     0.971 |                              0.923 |                              nan     |
| SANC − additive (attn pool)             | positivity   |                       0 |                     0.704 |              0.445 |                   0.007 |                      0.004 |                      0.01  |                       0.005 |                              0.009 |                   0.001 |                          0.032 |                                   1 |                       0.909 |                     0.97  |                              0.941 |                              nan     |
| Stain-only MIL (no deep emb)            | positivity   |                       0 |                     0.723 |              0.446 |                   0.006 |                      0.003 |                      0.009 |                       0.002 |                              0.007 |                   0.001 |                          0.028 |                                   1 |                       0.925 |                     0.969 |                              0.919 |                              nan     |
| Grad-CAM CNN (per-image normalised CAM) | positivity   |                       0 |                     0.25  |              0.323 |                   0.001 |                      0.001 |                      0.002 |                     nan     |                            nan     |                   0     |                          0     |                                   1 |                       0.057 |                     0.989 |                              0.357 |                              nan     |
| Grad-CAM CNN (CAM × P(HER2+))           | positivity   |                       0 |                     0.23  |              0.49  |                   0     |                      0     |                      0.001 |                     nan     |                            nan     |                   0     |                          0     |                                   1 |                       0.057 |                     0.996 |                              0.358 |                              nan     |
| SANC-MIL (proposed) [phikon]            | positivity   |                       0 |                     0.695 |              0.377 |                   0.008 |                      0.005 |                      0.013 |                       0.003 |                              0.014 |                   0.003 |                          0.063 |                                   1 |                       0.829 |                     0.958 |                              0.704 |                              nan     |
| SANC − stain features [phikon]          | positivity   |                       0 |                     0.66  |              0.364 |                   0.009 |                      0.005 |                      0.013 |                       0.003 |                              0.02  |                   0.003 |                          0.063 |                                   1 |                       0.829 |                     0.955 |                              0.655 |                              nan     |
| ABMIL (Ilse; BJC-2023) [phikon]         | attention    |                       0 |                     0.553 |              0.245 |                 nan     |                    nan     |                    nan     |                     nan     |                            nan     |                 nan     |                        nan     |                                 nan |                     nan     |                   nan     |                            nan     |                              nan     |
| MaxPool [phikon]                        | positivity   |                       0 |                     0.628 |              0.321 |                   0.001 |                      0     |                      0.001 |                       0.001 |                              0.002 |                   0     |                          0.007 |                                   1 |                       0.709 |                     0.952 |                              0.7   |                              nan     |

Variability over 3 repeated CV seeds (mean, SD):

```
                           test_AUROC         OOF_AUROC         IHC0_tile_FP_rate@0.5         OOF_IHC0_tile_FP_rate@0.5         IHC0_cores_with_any_tile>0.9         IHC3_cores_detected(any tile>0.5)      tile_AUROC_IHC3_vs_IHC0         fold_stability_spearman        
                                 mean     std      mean     std                  mean     std                      mean     std                         mean     std                              mean  std                    mean     std                    mean     std
method                                                                                                                                                                                                                                                                     
ABMIL (Ilse; BJC-2023)         0.9466  0.0013    0.9395  0.0028                   NaN     NaN                       NaN     NaN                          NaN     NaN                               NaN  NaN                     NaN     NaN                  0.5298  0.0518
MaxPool                        0.9779  0.0004    0.9690  0.0002                0.0006  0.0000                    0.0004  0.0000                       0.0000  0.0000                               1.0  0.0                  0.9323  0.0001                  0.8292  0.0044
SANC − negative constraint     0.9920  0.0015    0.9710  0.0039                0.0824  0.0146                    0.1111  0.0288                       0.2723  0.0781                               1.0  0.0                  0.9600  0.0035                  0.4753  0.2905
SANC − stain features          0.9603  0.0029    0.9573  0.0003                0.0127  0.0005                    0.0072  0.0001                       0.0329  0.0041                               1.0  0.0                  0.9444  0.0003                  0.8397  0.0033
SANC-MIL (proposed)            0.9931  0.0003    0.9787  0.0032                0.0054  0.0006                    0.0042  0.0008                       0.0293  0.0020                               1.0  0.0                  0.9691  0.0014                  0.6434  0.0885
```

### H1: paired per-core tests of IHC-0 false-positive tile fraction (A − B < 0 favours A)

| A                     | B                                       |   seed | split   |   n_cores |   mean_A |   mean_B |    diff |      lo |      hi |   wilcoxon_p |   cores_A_better |   cores_B_better |
|:----------------------|:----------------------------------------|-------:|:--------|----------:|---------:|---------:|--------:|--------:|--------:|-------------:|-----------------:|-----------------:|
| SANC-MIL (proposed)   | SANC − negative constraint              |      0 | test    |       284 |   0.0047 |   0.0678 | -0.0631 | -0.0805 | -0.0488 |       0      |              173 |                0 |
| SANC-MIL (proposed)   | SANC − negative constraint              |      0 | oof     |      1326 |   0.0041 |   0.0846 | -0.0805 | -0.0858 | -0.075  |       0      |              914 |                0 |
| SANC-MIL (proposed)   | SANC − negative constraint              |      1 | test    |       284 |   0.0054 |   0.098  | -0.0925 | -0.1063 | -0.0803 |       0      |              253 |                0 |
| SANC-MIL (proposed)   | SANC − negative constraint              |      1 | oof     |      1326 |   0.0031 |   0.1039 | -0.1008 | -0.106  | -0.0956 |       0      |             1002 |                0 |
| SANC-MIL (proposed)   | SANC − negative constraint              |      2 | test    |       284 |   0.0044 |   0.0661 | -0.0617 | -0.0813 | -0.0454 |       0      |              124 |                0 |
| SANC-MIL (proposed)   | SANC − negative constraint              |      2 | oof     |      1326 |   0.0038 |   0.1512 | -0.1474 | -0.1578 | -0.1374 |       0      |              893 |                7 |
| SANC − stain features | SANC − negative constraint              |      0 | test    |       284 |   0.0138 |   0.0678 | -0.054  | -0.0691 | -0.0407 |       0      |              166 |                4 |
| SANC − stain features | SANC − negative constraint              |      0 | oof     |      1326 |   0.0063 |   0.0846 | -0.0782 | -0.0836 | -0.0726 |       0      |              900 |               22 |
| SANC − stain features | SANC − negative constraint              |      1 | test    |       284 |   0.0147 |   0.098  | -0.0832 | -0.0952 | -0.072  |       0      |              249 |                2 |
| SANC − stain features | SANC − negative constraint              |      1 | oof     |      1326 |   0.0066 |   0.1039 | -0.0973 | -0.1024 | -0.0924 |       0      |              991 |               17 |
| SANC − stain features | SANC − negative constraint              |      2 | test    |       284 |   0.014  |   0.0661 | -0.0521 | -0.069  | -0.0379 |       0      |              116 |                5 |
| SANC − stain features | SANC − negative constraint              |      2 | oof     |      1326 |   0.0064 |   0.1512 | -0.1448 | -0.1553 | -0.1352 |       0      |              900 |               18 |
| SANC-MIL (proposed)   | MaxPool                                 |      0 | test    |       284 |   0.0047 |   0.0004 |  0.0043 |  0.0026 |  0.0063 |       0.0019 |                1 |               30 |
| SANC-MIL (proposed)   | MaxPool                                 |      0 | oof     |      1326 |   0.0041 |   0.0003 |  0.0038 |  0.0026 |  0.0053 |       0      |                0 |               88 |
| SANC-MIL (proposed)   | MaxPool                                 |      1 | test    |       284 |   0.0054 |   0.0005 |  0.0049 |  0.003  |  0.0071 |       0.0007 |                1 |               33 |
| SANC-MIL (proposed)   | MaxPool                                 |      1 | oof     |      1326 |   0.0031 |   0.0003 |  0.0029 |  0.0021 |  0.0038 |       0      |                0 |               96 |
| SANC-MIL (proposed)   | MaxPool                                 |      2 | test    |       284 |   0.0044 |   0.0005 |  0.0039 |  0.0022 |  0.0061 |       0.0155 |                0 |               22 |
| SANC-MIL (proposed)   | MaxPool                                 |      2 | oof     |      1326 |   0.0038 |   0.0003 |  0.0035 |  0.0026 |  0.0047 |       0      |                0 |              119 |
| SANC-MIL (proposed)   | Grad-CAM CNN (per-image normalised CAM) |      0 | test    |       284 |   0.0047 |   0.0013 |  0.0034 |  0.0017 |  0.0053 |       0.0417 |                9 |               28 |

## Segmentation student (agreement with PSEUDO-labels on held-out cores — not accuracy)

| model                                         |   same-budget agree_Dice |   same-budget agree_IoU |   same-budget agree_precision |   same-budget agree_recall |   same-budget agree_Dice_highconf |   same-budget evidence_AUROC_vs_pseudo |   final_epochs |   final agree_Dice |
|:----------------------------------------------|-------------------------:|------------------------:|------------------------------:|---------------------------:|----------------------------------:|---------------------------------------:|---------------:|-------------------:|
| MSA-U-Net-TR + ignore + confidence (proposed) |                    0.838 |                   0.72  |                         0.854 |                      0.821 |                             0.967 |                                  0.995 |             10 |              0.875 |
| U-Net + ignore + confidence                   |                    0.835 |                   0.716 |                         0.782 |                      0.894 |                             0.984 |                                  0.995 |              3 |              0.848 |
| MSA-U-Net-TR + naive hard labels              |                    0.827 |                   0.705 |                         0.91  |                      0.757 |                             0.919 |                                  0.993 |              4 |              0.827 |
| MSA-U-Net-TR + ignore only                    |                    0.856 |                   0.748 |                         0.859 |                      0.852 |                             0.962 |                                  0.995 |              4 |              0.856 |

Test-set proxies:

| model                                         |   IHC0_membranous_area_ge_mod |   IHC3_membranous_area_ge_mod |   mean_TTA_SD |   IHC3_to_IHC0_contrast |
|:----------------------------------------------|------------------------------:|------------------------------:|--------------:|------------------------:|
| MSA-U-Net-TR + ignore + confidence (proposed) |                        0.0003 |                        0.1686 |        0.0016 |                 563.571 |
| U-Net + ignore + confidence                   |                        0.0004 |                        0.1805 |        0.0016 |                 480.842 |
| MSA-U-Net-TR + naive hard labels              |                        0.0002 |                        0.1243 |        0.0018 |                 579.641 |
| MSA-U-Net-TR + ignore only                    |                        0.0004 |                        0.1728 |        0.0015 |                 492.442 |

## Clinical consistency of segmentation-derived features

| feature_set                        |   n_features |   test_AUROC |    lo |    hi |
|:-----------------------------------|-------------:|-------------:|------:|------:|
| rule membrane fractions            |            3 |        0.984 | 0.961 | 0.999 |
| student membrane fractions         |            3 |        0.987 | 0.969 | 0.999 |
| student + cell morphometry         |           10 |        0.988 | 0.973 | 0.998 |
| rule + cell morphometry            |           10 |        0.982 | 0.959 | 0.997 |
| Stain intensity rule (Pisula 2023) |            1 |        0.97  | 0.94  | 0.992 |

| split   | feature                 |   spearman_rho |   rho_lo |   rho_hi |   kruskal_p |   median_0 |   median_1 |   median_2- |   median_2+ |   median_3 |
|:--------|:------------------------|---------------:|---------:|---------:|------------:|-----------:|-----------:|------------:|------------:|-----------:|
| train   | rule_mem_mod_frac       |          0.588 |    0.546 |    0.624 |           0 |      0     |      0     |       0.002 |       0.008 |      0.064 |
| train   | stu_mem_mod_frac        |          0.601 |    0.556 |    0.637 |           0 |      0     |      0.001 |       0.004 |       0.015 |      0.127 |
| train   | stu_mem_strong_frac     |          0.687 |    0.645 |    0.725 |           0 |      0     |      0     |       0     |       0.001 |      0.045 |
| train   | stu_cell_cov>=0.5       |          0.811 |    0.772 |    0.849 |           0 |      0     |      0     |       0     |       0.023 |      0.122 |
| train   | stu_cell_cov>=0.5&mod   |          0.775 |    0.729 |    0.818 |           0 |      0     |      0     |       0     |       0.001 |      0.081 |
| train   | stu_cell_median_cov_pos |          0.562 |    0.521 |    0.6   |           0 |      0     |      0.079 |       0.087 |       0.2   |      0.331 |
| test    | rule_mem_mod_frac       |          0.45  |    0.374 |    0.526 |           0 |      0     |    nan     |       0.011 |       0.024 |      0.08  |
| test    | stu_mem_mod_frac        |          0.451 |    0.373 |    0.526 |           0 |      0     |    nan     |       0.021 |       0.042 |      0.15  |
| test    | stu_mem_strong_frac     |          0.487 |    0.402 |    0.565 |           0 |      0     |    nan     |       0.001 |       0.005 |      0.071 |
| test    | stu_cell_cov>=0.5       |          0.576 |    0.465 |    0.663 |           0 |      0     |    nan     |       0.073 |       0.041 |      0.151 |
| test    | stu_cell_cov>=0.5&mod   |          0.766 |    0.628 |    0.877 |           0 |      0     |    nan     |       0.001 |       0.006 |      0.072 |
| test    | stu_cell_median_cov_pos |          0.436 |    0.352 |    0.512 |           0 |      0.056 |    nan     |       0.252 |       0.202 |      0.327 |

### H3: single-feature separation of 3+ vs 2+ and of 2+/FISH+ vs 2+/FISH− (direction fixed on train; top 5 per block)

| comparison           | split   | feature                 | direction   |   n_pos |   n_neg |   AUROC |    lo |    hi |   p_mannwhitney |   p_holm |
|:---------------------|:--------|:------------------------|:------------|--------:|--------:|--------:|------:|------:|----------------:|---------:|
| 2+/FISH+ vs 2+/FISH- | train   | stu_cell_median_cov_pos | +           |      56 |      17 |   0.922 | 0.856 | 0.978 |           0     |    0     |
| 2+/FISH+ vs 2+/FISH- | train   | stu_cell_cov>=0.3       | +           |      56 |      17 |   0.793 | 0.677 | 0.894 |           0     |    0.005 |
| 2+/FISH+ vs 2+/FISH- | train   | stu_cell_cov>=0.5       | +           |      56 |      17 |   0.784 | 0.676 | 0.877 |           0     |    0.006 |
| 2+/FISH+ vs 2+/FISH- | train   | rule_cell_cov>=0.3      | +           |      56 |      17 |   0.773 | 0.653 | 0.879 |           0.001 |    0.012 |
| 2+/FISH+ vs 2+/FISH- | train   | rule_cell_cov>=0.5      | +           |      56 |      17 |   0.762 | 0.65  | 0.86  |           0.001 |    0.013 |
| 2+/FISH+ vs 2+/FISH- | test    | rule_mem_strong_frac    | +           |       8 |       7 |   0.839 | 0.625 | 1     |           0.029 |    0.578 |
| 2+/FISH+ vs 2+/FISH- | test    | stu_mem_strong_frac     | +           |       8 |       7 |   0.821 | 0.571 | 1     |           0.04  |    0.762 |
| 2+/FISH+ vs 2+/FISH- | test    | stu_mem_mod_frac        | +           |       8 |       7 |   0.732 | 0.464 | 0.946 |           0.152 |    1     |
| 2+/FISH+ vs 2+/FISH- | test    | rule_mem_mod_frac       | +           |       8 |       7 |   0.732 | 0.464 | 0.946 |           0.152 |    1     |
| 2+/FISH+ vs 2+/FISH- | test    | stu_cell_cov>=0.5&mod   | +           |       8 |       7 |   0.732 | 0.429 | 0.982 |           0.148 |    1     |
| 3+ vs 2+             | train   | rule_mem_strong_frac    | +           |     114 |      73 |   0.885 | 0.838 | 0.931 |           0     |    0     |
| 3+ vs 2+             | train   | stu_mem_strong_frac     | +           |     114 |      73 |   0.883 | 0.836 | 0.93  |           0     |    0     |
| 3+ vs 2+             | train   | rule_mem_mod_frac       | +           |     114 |      73 |   0.838 | 0.781 | 0.893 |           0     |    0     |
| 3+ vs 2+             | train   | stu_mem_mod_frac        | +           |     114 |      73 |   0.835 | 0.779 | 0.892 |           0     |    0     |
| 3+ vs 2+             | train   | stu_cell_cov>=0.5&mod   | +           |     114 |      73 |   0.829 | 0.77  | 0.884 |           0     |    0     |
| 3+ vs 2+             | test    | stu_mem_strong_frac     | +           |       8 |      15 |   1     | 1     | 1     |           0     |    0     |
| 3+ vs 2+             | test    | rule_mem_strong_frac    | +           |       8 |      15 |   1     | 1     | 1     |           0     |    0     |
| 3+ vs 2+             | test    | stu_mem_mod_frac        | +           |       8 |      15 |   0.967 | 0.883 | 1     |           0     |    0.001 |
| 3+ vs 2+             | test    | rule_mem_mod_frac       | +           |       8 |      15 |   0.967 | 0.883 | 1     |           0     |    0.001 |
| 3+ vs 2+             | test    | stu_cell_cov>=0.5&mod   | +           |       8 |      15 |   0.967 | 0.875 | 1     |           0     |    0.005 |

## Computational cost

| component                         | model                                         |   trainable_params |
|:----------------------------------|:----------------------------------------------|-------------------:|
| MIL head                          | MeanPool                                      |             262657 |
| MIL head                          | MaxPool                                       |             262914 |
| MIL head                          | ABMIL (Ilse; BJC-2023)                        |             328578 |
| MIL head                          | CLAM-SB                                       |             329606 |
| MIL head                          | TransMIL-lite                                 |            1317121 |
| MIL head                          | SANC-MIL (proposed)                           |             346438 |
| MIL head                          | SANC − stain features                         |             329094 |
| MIL head                          | SANC − negative constraint                    |             346438 |
| MIL head                          | SANC − top-k constraint                       |             346438 |
| MIL head                          | SANC − ordinal head                           |             346179 |
| MIL head                          | SANC − additive (attn pool)                   |             346438 |
| MIL head                          | Stain-only MIL (no deep emb)                  |              84294 |
| segmentation student              | MSA-U-Net-TR + ignore + confidence (proposed) |           25899574 |
| segmentation student              | U-Net + ignore + confidence                   |           24446002 |
| segmentation student              | MSA-U-Net-TR + naive hard labels              |           25899574 |
| segmentation student              | MSA-U-Net-TR + ignore only                    |           25899574 |
| CNN baseline (Grad-CAM)           | resnet18.tv_in1k                              |           11177025 |
| frozen tile encoder (not trained) | timm:resnet50.tv_in1k                         |            8543296 |

Whole-core test segmentation: {'n_cores': 307, 'median_s': 2.619821071624756, 'q25_s': 2.5633383989334106, 'q75_s': 2.6746034622192383, 'tta': 4, 'gpu': 'Tesla T4'}

| section                                                   |   minutes |   sessions |   max_RAM_GB |   peak_GPU_GB |   output_folder_GB |
|:----------------------------------------------------------|----------:|-----------:|-------------:|--------------:|-------------------:|
| pass 1 featurisation                                      |      32.5 |          2 |          2.7 |           3.2 |                4.3 |
| MIL training (variants, leakage, seeds, encoder ablation) |      50.8 |          2 |          3.9 |           1.5 |                4.3 |
| Grad-CAM baseline                                         |      54.2 |          2 |          4.3 |           6.4 |                4.3 |
| classification evaluation + uncertainty                   |      11.8 |          2 |          4.3 |           0.8 |                4.3 |
| localisation analysis                                     |       0.4 |          2 |          4.3 |           0.8 |                4.3 |
| segmentation students                                     |     165.2 |          1 |          3.8 |           8.3 |                4   |
| core & cell features                                      |      66.6 |          1 |          4.3 |           5.4 |                4.2 |
| whole-core test segmentation                              |      16.5 |          1 |          4.3 |           6   |                4.2 |
| clinical consistency                                      |       4.1 |          1 |          4.3 |           3.6 |                4.2 |

## Claims ledger

**Supported by this run (with the stated uncertainty):**
- Image-level HER2 status on the independent test set: SANC-MIL AUROC 0.993 (95% CI 0.984–0.999); ABMIL 0.948 (95% CI 0.873–0.995); stain-intensity rule 0.970 (95% CI 0.940–0.992).
- Instance positivity on IHC-0 test cores (label-grounded specificity), seed mean ± SD: MaxPool: FP@0.5 0.001 ± 0.000; SANC − negative constraint: FP@0.5 0.082 ± 0.015; SANC − stain features: FP@0.5 0.013 ± 0.001; SANC-MIL (proposed): FP@0.5 0.005 ± 0.001.
- Grad-CAM baseline (per-image normalised, as in the original draft): IHC-0 tile FP@0.5 = 0.001.
- ISH routing (threshold tuned on OOF for 80% sensitivity): test sensitivity for IHC 2 = 0.93, routing rate in non-IHC 2 = 0.16.
- H1 (negative-bag constraint reduces IHC-0 false-positive tiles): SUPPORTED: in every seed and on both the test set and OOF train, the CI of the per-core FP difference lies below 0 (mean diff -0.091).
- H3 (membrane/cell features separate IHC 3+ from 2+): train-selected feature `rule_mem_strong_frac`: test AUROC 1.000 (95% CI 1.000–1.000, n=8 vs 15) — SUPPORTED (CI excludes 0.5).
- Classification differences between MIL variants: see Holm-corrected table; a difference is claimed only where p_holm < 0.05.
- Monotonic association of membrane-derived features with IHC score (see Spearman ρ / Kruskal–Wallis table).
- QC and leakage analyses: see qc_summary.json and leakage_sensitivity.csv.

**NOT established (do not claim):**
- Pixel-level segmentation accuracy of membranes, tumour regions or cells: no expert masks were available.
- That 'pseudo-label agreement' Dice equals accuracy — it measures reproduction of the training signal.
- That attention / evidence maps mark the cells a pathologist would score — they mark what drives the model.
- Any IHC 1+ test performance (the test set has no 1+ cores); IHC 2+ results rest on n≈15 cores.
- That the model can replace FISH for IHC 2+: equivocal cores are routed to ISH by design.
- Generalisation across institutions/scanners: single centre, single scanner here.

Next step for validated segmentation: annotate `annotation_worklist/` (132 ROIs, 44 cores; expected Dice CI ±0.04).