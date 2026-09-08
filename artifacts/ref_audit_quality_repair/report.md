# Reference Segment Audit

**Conclusion: NO-LEAK.**
Checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024/checkpoints/final`

## Position IDs

- disjoint: `True`
- overlaps: `{'image_garment': 0, 'image_hair': 0, 'garment_hair': 0}`
- image: N=3072, y=[0,63], x=[0,47]
- garment_reference: N=3072, y=[0,63], x=[64,111]
- hair_reference: N=768, y=[64,126], x=[0,46]

## ControlNet Residuals

Maximum RMS on either reference segment: `0.000e+00`.
The required invariant is exact zero; ControlNet may affect only the first 3072 image tokens.

## Reference Content

Maximum decoded far-outside changed-pixel ratio: `0.0001`.
The changed-pixel test compares each VAE decode against its intended masked canvas. An independently encoded all-gray canvas is not used because it is an out-of-distribution VAE input and produced a severe false color cast in the audit smoke test.

| sample | route | area | outer 2px changed | far outside changed | far outside MAD |
|---|---|---:|---:|---:|---:|
| 00041 | garment | 0.2654 | 0.0561 | 0.0000 | 2.009 |
| 00041 | hair | 0.0253 | 0.1897 | 0.0000 | 2.004 |
| 00430 | garment | 0.2819 | 0.0246 | 0.0000 | 1.211 |
| 00430 | hair | 0.0206 | 0.1991 | 0.0000 | 2.016 |
| 00096 | garment | 0.1948 | 0.0543 | 0.0000 | 2.139 |
| 00096 | hair | 0.0123 | 0.1564 | 0.0000 | 2.123 |
| 00093 | garment | 0.1716 | 0.2384 | 0.0000 | 2.025 |
| 00093 | hair | 0.0234 | 0.1506 | 0.0000 | 2.008 |
| 00571 | garment | 0.3271 | 0.3140 | 0.0001 | 1.723 |
| 00571 | hair | 0.0095 | 0.2404 | 0.0000 | 2.015 |
| 00807 | garment | 0.2174 | 0.2925 | 0.0000 | 1.811 |
| 00807 | hair | 0.0423 | 0.1499 | 0.0000 | 2.004 |
| 00126 | garment | 0.0986 | 0.1541 | 0.0000 | 2.057 |
| 00126 | hair | 0.0109 | 0.1266 | 0.0000 | 2.032 |
| 00654 | garment | 0.2075 | 0.0497 | 0.0000 | 2.050 |
| 00654 | hair | 0.0222 | 0.1667 | 0.0000 | 2.022 |

## Neutral Background Audit

- configured reference gray: `RGB(227,227,227)`
- mannequin parsing-background median: `[228.0, 227.0, 226.0]`
- mannequin parsing-background mean: `[229.14, 227.6, 225.88]`
- maximum channel difference from background median: `1.00`
- interpretation: the reference canvas is aligned with the training-image background statistic.

## Cross-Segment Attention

Values are exact softmax mass averaged over heads, selected layers, samples, and tau values.

| image query region | text | image/self | garment ref | hair ref | sum |
|---|---:|---:|---:|---:|---:|
| background | 0.0784 | 0.6910 | 0.1837 | 0.0468 | 1.0000 |
| cloth | 0.0774 | 0.6915 | 0.1836 | 0.0476 | 1.0000 |
| face | 0.0829 | 0.6882 | 0.1790 | 0.0499 | 1.0000 |
| hair | 0.0803 | 0.6892 | 0.1804 | 0.0501 | 1.0000 |

Background total reference attention: `0.2306`.
Foreground-region mean reference attention: `0.2302`.
Background / foreground attention enrichment: `1.0019`.
Raw reference mass is diagnostic only because it scales with the 3840 appended keys; it becomes a leakage reason only when the neutral canvas is mismatched or background queries are enriched relative to cloth/face/hair queries.

## Decision

- conclusion: **NO-LEAK**
- reasons: `['none']`
- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.
- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.
