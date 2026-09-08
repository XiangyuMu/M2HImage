# Spatial sim_target Failure Attribution

**Conclusion: CAPACITY-OR-TRAINING.**
**sleeve_mismatch overall rate: 36.75% (147/400).**

Run: `spatial_hair_ab_space`; images: `400`; valid sim_target: `399`.
Quantiles over valid rows: P25=`0.379771`, P75=`0.546048`. Detector failures are assigned to the low group.
Deterministic panel seed: `20260824`.

## Failure Rates By sim_target Group

| label | low | middle | high | low/high | point-biserial r | p |
|---|---:|---:|---:|---:|---:|---:|
| `face_occluded_by_hair` | 74.26% (75/101) | 67.84% (135/199) | 47.00% (47/100) | 1.58 | -0.0715 | 0.154146 |
| `face_undetected_or_lowconf` | 0.99% (1/101) | 0.00% (0/199) | 0.00% (0/100) | inf | N/A | N/A |
| `hair_overgrown` | 21.78% (22/101) | 53.27% (106/199) | 72.00% (72/100) | 0.30 | 0.3221 | 0.000000 |
| `accessory_leak` | 55.45% (56/101) | 51.76% (103/199) | 38.00% (38/100) | 1.46 | -0.0576 | 0.250834 |
| `sleeve_mismatch` | 35.64% (36/101) | 34.67% (69/199) | 42.00% (42/100) | 0.85 | 0.0409 | 0.415151 |

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
