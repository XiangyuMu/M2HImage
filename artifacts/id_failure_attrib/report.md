# Spatial sim_target Failure Attribution

**Conclusion: CAPACITY-OR-TRAINING.**
**sleeve_mismatch overall rate: 35.50% (142/400).**

Run: `spatial_quality_repair`; images: `400`; valid sim_target: `398`.
Quantiles over valid rows: P25=`0.301493`, P75=`0.495166`. Detector failures are assigned to the low group.
Deterministic panel seed: `20260824`.

## Failure Rates By sim_target Group

| label | low | middle | high | low/high | point-biserial r | p |
|---|---:|---:|---:|---:|---:|---:|
| `face_occluded_by_hair` | 72.55% (74/102) | 70.71% (140/198) | 70.00% (70/100) | 1.04 | -0.0206 | 0.682244 |
| `face_undetected_or_lowconf` | 2.94% (3/102) | 0.00% (0/198) | 0.00% (0/100) | inf | -0.1153 | 0.021388 |
| `hair_overgrown` | 28.43% (29/102) | 58.08% (115/198) | 72.00% (72/100) | 0.39 | 0.3274 | 0.000000 |
| `accessory_leak` | 46.08% (47/102) | 48.99% (97/198) | 35.00% (35/100) | 1.32 | -0.1175 | 0.019055 |
| `sleeve_mismatch` | 31.37% (32/102) | 35.86% (71/198) | 39.00% (39/100) | 0.80 | 0.0593 | 0.237614 |

The preregistered hair-dominant rule is low-group rate >= 2x high-group rate for either `face_occluded_by_hair` or `hair_overgrown`.
Triggered hair labels: `none`.

## Deterministic Qualitative Grids

Each row is `[mannequin | reference person | generated | enlarged generated-face crop]`.

- [low sim_target](sim_low_grid.png)
- [middle sim_target](sim_middle_grid.png)
- [high sim_target](sim_high_grid.png)

## Definitions And Limitations

- `face_occluded_by_hair`: hair occupies more than 25% of a DWPose/parsing-derived geometric face zone. FASHN labels are mutually exclusive, so direct face-mask/hair-mask intersection would be identically zero.
- `face_undetected_or_lowconf`: held-out runner detection failed or RetinaFace confidence is below 0.60.
- `hair_overgrown`: generated/reference full-frame hair-area ratio exceeds 1.50.
- `accessory_leak` is explicitly heuristic: small jewelry/glasses or high-saturation components in reference/generated ear bands must have a similar hue histogram. It is not treated as a causal detector.
- `sleeve_mismatch`: the absolute change in parsed arms/hands occupancy inside source-DWPose elbow-to-wrist capsules exceeds 0.15.
- Existing generated parsing, DWPose predictions, and held-out identity CSV are reused read-only; no image generation or training occurs.

Machine-readable outputs: [per_image.csv](per_image.csv), [cross_table.csv](cross_table.csv), [summary.json](summary.json).
