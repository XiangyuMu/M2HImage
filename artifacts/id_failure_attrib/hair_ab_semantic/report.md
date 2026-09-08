# Spatial sim_target Failure Attribution

**Conclusion: CAPACITY-OR-TRAINING.**
**sleeve_mismatch overall rate: 35.50% (142/400).**

Run: `spatial_hair_ab_semantic`; images: `400`; valid sim_target: `400`.
Quantiles over valid rows: P25=`0.403258`, P75=`0.564373`. Detector failures are assigned to the low group.
Deterministic panel seed: `20260824`.

## Failure Rates By sim_target Group

| label | low | middle | high | low/high | point-biserial r | p |
|---|---:|---:|---:|---:|---:|---:|
| `face_occluded_by_hair` | 82.00% (82/100) | 79.00% (158/200) | 74.00% (74/100) | 1.11 | -0.0370 | 0.461109 |
| `face_undetected_or_lowconf` | 0.00% (0/100) | 0.00% (0/200) | 0.00% (0/100) | 1.00 | N/A | N/A |
| `hair_overgrown` | 49.00% (49/100) | 66.00% (132/200) | 77.00% (77/100) | 0.64 | 0.1791 | 0.000319 |
| `accessory_leak` | 62.00% (62/100) | 43.50% (87/200) | 40.00% (40/100) | 1.55 | -0.1556 | 0.001799 |
| `sleeve_mismatch` | 35.00% (35/100) | 34.50% (69/200) | 38.00% (38/100) | 0.92 | 0.0370 | 0.461144 |

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
