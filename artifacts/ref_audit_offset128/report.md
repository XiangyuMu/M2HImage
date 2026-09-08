# Reference Segment Audit

**Conclusion: LEAK-FOUND.**
Checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024/checkpoints/final`

## Position IDs

- disjoint: `True`
- overlaps: `{'image_garment': 0, 'image_hair': 0, 'garment_hair': 0}`
- image: N=3072, y=[0,63], x=[0,47]
- garment_reference: N=3072, y=[0,63], x=[128,175]
- hair_reference: N=768, y=[128,190], x=[0,46]

## ControlNet Residuals

Maximum RMS on either reference segment: `0.000e+00`.
The required invariant is exact zero; ControlNet may affect only the first 3072 image tokens.

## Reference Content

Maximum decoded far-outside changed-pixel ratio: `0.0000`.
The changed-pixel test compares each VAE decode against its intended masked canvas. An independently encoded all-gray canvas is not used because it is an out-of-distribution VAE input and produced a severe false color cast in the audit smoke test.

| sample | route | area | outer 2px changed | far outside changed | far outside MAD |
|---|---|---:|---:|---:|---:|
| 00041 | garment | 0.2654 | 0.0561 | 0.0000 | 2.009 |
| 00041 | hair | 0.0253 | 0.1897 | 0.0000 | 2.004 |
| 00430 | garment | 0.2819 | 0.0246 | 0.0000 | 1.211 |
| 00430 | hair | 0.0206 | 0.1991 | 0.0000 | 2.016 |

## Neutral Background Audit

- configured reference gray: `RGB(227,227,227)`
- mannequin parsing-background median: `[231.0, 231.0, 231.0]`
- mannequin parsing-background mean: `[230.8, 230.36, 230.38]`
- maximum channel difference from background median: `4.00`
- interpretation: the reference canvas is aligned with the training-image background statistic.

## Cross-Segment Attention

Values are exact softmax mass averaged over heads, selected layers, samples, and tau values.

| image query region | text | image/self | garment ref | hair ref | sum |
|---|---:|---:|---:|---:|---:|
| background | 0.0799 | 0.6904 | 0.1810 | 0.0488 | 1.0000 |
| cloth | 0.0792 | 0.6914 | 0.1804 | 0.0489 | 1.0000 |
| face | 0.0858 | 0.6901 | 0.1742 | 0.0499 | 1.0000 |
| hair | 0.0826 | 0.6911 | 0.1759 | 0.0504 | 1.0000 |

Background total reference attention: `0.2297`.

## Decision

- conclusion: **LEAK-FOUND**
- reasons: `['background queries allocate too much mass to reference segments (0.230 > 0.150)']`
- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.
- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.
