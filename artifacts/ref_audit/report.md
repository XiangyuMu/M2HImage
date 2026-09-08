# Reference Segment Audit

**Conclusion: LEAK-FOUND.**
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

Maximum decoded far-outside changed-pixel ratio: `0.0000`.
The changed-pixel test compares each VAE decode against its intended masked canvas. An independently encoded all-gray canvas is not used because it is an out-of-distribution VAE input and produced a severe false color cast in the audit smoke test.

| sample | route | area | outer 2px changed | far outside changed | far outside MAD |
|---|---|---:|---:|---:|---:|
| 00041 | garment | 0.2729 | 0.0522 | 0.0000 | 0.917 |
| 00041 | hair | 0.0292 | 0.2126 | 0.0000 | 0.335 |
| 00430 | garment | 0.2924 | 0.0602 | 0.0000 | 0.374 |
| 00430 | hair | 0.0239 | 0.1421 | 0.0000 | 0.336 |
| 00096 | garment | 0.2064 | 0.0513 | 0.0000 | 0.408 |
| 00096 | hair | 0.0149 | 0.1373 | 0.0000 | 0.340 |
| 00093 | garment | 0.1776 | 0.1514 | 0.0000 | 0.367 |
| 00093 | hair | 0.0279 | 0.1805 | 0.0000 | 0.339 |
| 00571 | garment | 0.3361 | 0.2013 | 0.0000 | 0.674 |
| 00571 | hair | 0.0109 | 0.1462 | 0.0000 | 0.516 |
| 00807 | garment | 0.2253 | 0.2039 | 0.0000 | 0.064 |
| 00807 | hair | 0.0464 | 0.1760 | 0.0000 | 0.352 |
| 00126 | garment | 0.1032 | 0.1235 | 0.0000 | 0.313 |
| 00126 | hair | 0.0123 | 0.1047 | 0.0000 | 0.342 |
| 00654 | garment | 0.2149 | 0.0302 | 0.0000 | 0.604 |
| 00654 | hair | 0.0261 | 0.1091 | 0.0000 | 0.389 |

## Neutral Background Audit

- configured reference gray: `RGB(127,127,127)`
- mannequin parsing-background median: `[228.0, 227.0, 226.0]`
- mannequin parsing-background mean: `[229.14, 227.6, 225.88]`
- interpretation: the current reference canvas is much darker than the training-image background and is globally visible through cross-segment attention.

## Cross-Segment Attention

Values are exact softmax mass averaged over heads, selected layers, samples, and tau values.

| image query region | text | image/self | garment ref | hair ref | sum |
|---|---:|---:|---:|---:|---:|
| background | 0.0750 | 0.7013 | 0.1777 | 0.0460 | 1.0000 |
| cloth | 0.0741 | 0.7027 | 0.1769 | 0.0462 | 1.0000 |
| face | 0.0789 | 0.7002 | 0.1742 | 0.0467 | 1.0000 |
| hair | 0.0766 | 0.7021 | 0.1747 | 0.0466 | 1.0000 |

Background total reference attention: `0.2237`.

## Decision

- conclusion: **LEAK-FOUND**
- reasons: `['background queries allocate too much mass to reference segments (0.224 > 0.150)']`
- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.
- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.
