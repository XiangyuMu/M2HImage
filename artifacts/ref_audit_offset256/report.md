# Reference Segment Audit

**Conclusion: LEAK-FOUND.**
Checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024/checkpoints/final`

## Position IDs

- disjoint: `True`
- overlaps: `{'image_garment': 0, 'image_hair': 0, 'garment_hair': 0}`
- image: N=3072, y=[0,63], x=[0,47]
- garment_reference: N=3072, y=[0,63], x=[256,304]
- hair_reference: N=768, y=[256,318], x=[0,46]

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
| background | 0.0799 | 0.6909 | 0.1812 | 0.0480 | 1.0000 |
| cloth | 0.0779 | 0.6917 | 0.1825 | 0.0479 | 1.0000 |
| face | 0.0835 | 0.6928 | 0.1750 | 0.0487 | 1.0000 |
| hair | 0.0810 | 0.6940 | 0.1762 | 0.0488 | 1.0000 |

Background total reference attention: `0.2292`.

## Decision

- conclusion: **LEAK-FOUND**
- reasons: `['background queries allocate too much mass to reference segments (0.229 > 0.150)']`
- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.
- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.
