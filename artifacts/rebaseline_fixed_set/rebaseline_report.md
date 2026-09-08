# Watcher Fixed-Set Rebaseline

Protocol: `40826872040bb9738a5b5f3b4e06b527f6d3e6a8e74d24db92326a144ec48517`; 16 fixed validation samples, four per garment type, all reference hair area >=1%.

## Decision Lines

- Fixed-set Garment-DINO median: step-2000 `0.789799` -> step-2500 `0.917740`; one-sided paired p=`1`; **GARMENT-OK**.
- Fixed-set head5 median: step-2000 `0.019803` -> step-2500 `0.018719`; one-sided paired p=`0.56987`; **HEAD-OK**.
- Lowest step-2500 skirt is `00093`; panel: [skirt_worst_00093_panel.png](skirt_worst_00093_panel.png). Human conclusion: **the parsing mask is correctly aligned. The generated image preserves the black top and pink tied lower garment, so this is not a parsing failure or garment collapse; remaining differences are local texture and construction fidelity**.
- Legacy 0.480 skirt is `18704`; panel: [legacy_048_skirt_18704_panel.png](legacy_048_skirt_18704_panel.png). Human conclusion: **the FASHN mask is correctly on the orange blouse and black skirt; this is not a parsing failure or garment collapse. The generated item is recognizably the same outfit, but its sleeve/button, pleat, and fabric detail are blurred or altered, so the low score is a real fidelity/domain outlier amplified by DINO**.

A worse median is called regressed only when the paired one-sided Wilcoxon test is significant at p<0.05.

## Aggregate Metrics

| metric | step 2000 mean | step 2000 median | step 2500 mean | step 2500 median |
|---|---:|---:|---:|---:|
| garment_dino | 0.748275 | 0.789799 | 0.871821 | 0.917740 |
| garment_lpips | 0.432906 | 0.413784 | 0.251067 | 0.188466 |
| hair_dino | 0.588968 | 0.603145 | 0.593015 | 0.622311 |
| hair_lab | 47.018407 | 43.994129 | 27.333840 | 21.651465 |
| body | 0.010330 | 0.010740 | 0.010624 | 0.009833 |
| head5 | 0.021220 | 0.019803 | 0.019499 | 0.018719 |

## Per-Type Medians

| garment type | Garment-DINO 2000 | Garment-DINO 2500 | head5 2000 | head5 2500 |
|---|---:|---:|---:|---:|
| top | 0.858095 | 0.925212 | 0.025207 | 0.012807 |
| dress | 0.763165 | 0.912149 | 0.019803 | 0.020236 |
| pants | 0.765874 | 0.908690 | 0.010736 | 0.007324 |
| skirt | 0.787748 | 0.896799 | 0.031893 | 0.032316 |

## Per-Sample Values

| id | type | garment 2000 | garment 2500 | delta | head5 2000 | head5 2500 | delta | outlier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 00041 | top | 0.705558 | 0.897885 | +0.192327 | 0.005006 | 0.004847 | -0.000159 |  |
| 00430 | dress | 0.246783 | 0.372215 | +0.125432 | 0.020638 | 0.019888 | -0.000750 |  |
| 00096 | pants | 0.756384 | 0.927907 | +0.171523 | 0.004661 | 0.004225 | -0.000435 |  |
| 00093 | skirt | 0.556583 | 0.764761 | +0.208178 | 0.029602 | 0.031341 | +0.001738 |  |
| 00571 | top | 0.851909 | 0.941462 | +0.089553 | 0.042905 | 0.009270 | -0.033635 | head |z|>2 |
| 00807 | dress | 0.848243 | 0.926517 | +0.078274 | 0.018969 | 0.020585 | +0.001616 |  |
| 00126 | pants | 0.616298 | 0.810922 | +0.194624 | 0.006309 | 0.009506 | +0.003198 |  |
| 00654 | skirt | 0.809008 | 0.934279 | +0.125271 | 0.011143 | 0.033291 | +0.022148 |  |
| 00621 | top | 0.864280 | 0.908962 | +0.044682 | 0.018594 | 0.022232 | +0.003638 |  |
| 02331 | dress | 0.678088 | 0.897780 | +0.219692 | 0.006979 | 0.007183 | +0.000203 |  |
| 00169 | pants | 0.930069 | 0.970947 | +0.040878 | 0.027717 | 0.017549 | -0.010168 |  |
| 00970 | skirt | 0.804235 | 0.970807 | +0.166572 | 0.034183 | 0.027160 | -0.007023 |  |
| 00664 | top | 0.871634 | 0.947855 | +0.076222 | 0.031821 | 0.016344 | -0.015477 |  |
| 02811 | dress | 0.886701 | 0.928037 | +0.041335 | 0.027908 | 0.042268 | +0.014360 |  |
| 00184 | pants | 0.775363 | 0.889473 | +0.114109 | 0.015162 | 0.005141 | -0.010021 |  |
| 01785 | skirt | 0.771260 | 0.859319 | +0.088059 | 0.037928 | 0.041157 | +0.003229 |  |

## Legacy Trajectory Audit

Legacy response_track first step: `2000`; protocol status: **legacy incompatible protocol**.
The legacy +0.089 Garment-DINO slope is not admitted as fixed-protocol evidence unless the stored protocol hash matches.

Raw per-image CSVs are under `step2000/metrics_v2/` and `step2500/metrics_v2/`.
