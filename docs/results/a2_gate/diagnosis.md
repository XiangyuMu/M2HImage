# A2 Diagnostic Report

- A2 run: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase2_a2_diff_r16_4000_768x1024`
- resolved config: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase2_a2_diff_r16_4000_768x1024/resolved_config.yaml`
- logged metric rows: 403

## Differential Loss Binding

- calibrated hinge_g: **0.029145**
- post-calibration loss_pair mean: 0.290075
- post-calibration loss_teach mean: 0.151429
- post-calibration loss_inv mean: 0.003756
- post-calibration loss_hinge mean: 0.001000
- weighted differential / loss_pair: 26.38%
- hinge activation mean/max: 12.89% / 66.67%
- face differential norm start/end: 0.031812 / 0.036465
- curves: [diagnosis_training_curves.png](diagnosis_training_curves.png)

**Binding conclusion: BOUND**

## Paired Held-out DeltaID Gain

- paired images: 400
- A2 mean: 0.420142
- B2-cont mean: 0.408815
- mean/median gain: 0.011327 / 0.012331
- Wilcoxon greater p: 0.000002647
- rank-biserial effect: 0.262693
- gain P10/P25/P75/P90: -0.055400 / -0.017202 / 0.037817 / 0.076997
- histogram: [diagnosis_deltaid_gain.png](diagnosis_deltaid_gain.png)

**DeltaID conclusion: SIGNIFICANT**

## Residual Garment Instability

- paired mannequin IDs: 50
- bottom-quartile size: 13
- low-tail overlap: 7
- low-tail Jaccard: 0.3684

| garment type | count | A2 mean | B2-cont mean | gain | greater p |
|---|---:|---:|---:|---:|---:|
| dress | 13 | 0.8804 | 0.8787 | +0.0017 | 0.419678 |
| pants | 13 | 0.9105 | 0.9011 | +0.0094 | 0.169800 |
| skirt | 12 | 0.9041 | 0.8913 | +0.0128 | 0.046143 |
| top | 12 | 0.9123 | 0.9101 | +0.0022 | 0.338623 |

A2 low-tail mids: `['01264', '01968', '02725', '05214', '16049', '17881', '20743', '29153', '32740', '38609', '41344', '41915', '47243']`

B2-cont low-tail mids: `['00515', '01203', '01264', '01968', '02725', '16049', '17881', '32740', '34195', '35002', '38609', '41016', '43313']`

## Preregistered Diagnostic Decision

**PROCEED: 转锚地基成立，按计划进入 A4。**
