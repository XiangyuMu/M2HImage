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

Maximum decoded far-outside changed-pixel ratio: `1.0000`.
The changed-pixel test compares each VAE decode against its intended masked canvas. An independently encoded all-gray canvas is not used because it is an out-of-distribution VAE input and produced a severe false color cast in the audit smoke test.

| sample | route | area | outer 2px changed | far outside changed | far outside MAD |
|---|---|---:|---:|---:|---:|
| 00041 | garment | 0.2654 | 0.7641 | 1.0000 | 100.905 |
| 00041 | hair | 0.0253 | 0.8836 | 1.0000 | 100.335 |
| 00430 | garment | 0.2819 | 0.6863 | 0.9999 | 100.357 |
| 00430 | hair | 0.0206 | 0.9245 | 1.0000 | 100.330 |
| 00096 | garment | 0.1948 | 0.7899 | 1.0000 | 100.402 |
| 00096 | hair | 0.0123 | 0.8652 | 1.0000 | 100.337 |
| 00093 | garment | 0.1716 | 0.9370 | 1.0000 | 100.355 |
| 00093 | hair | 0.0234 | 0.8010 | 1.0000 | 100.332 |

## Neutral Background Audit

- configured reference gray: `RGB(127,127,127)`
- mannequin parsing-background median: `[230.0, 230.0, 230.0]`
- mannequin parsing-background mean: `[227.07, 226.83, 226.34]`
- interpretation: the current reference canvas is much darker than the training-image background and is globally visible through cross-segment attention.

## Cross-Segment Attention

Values are exact softmax mass averaged over heads, selected layers, samples, and tau values.

| image query region | text | image/self | garment ref | hair ref | sum |
|---|---:|---:|---:|---:|---:|
| background | 0.0763 | 0.7022 | 0.1767 | 0.0447 | 1.0000 |
| cloth | 0.0757 | 0.7024 | 0.1767 | 0.0451 | 1.0000 |
| face | 0.0803 | 0.6994 | 0.1742 | 0.0461 | 1.0000 |
| hair | 0.0784 | 0.7024 | 0.1736 | 0.0456 | 1.0000 |

Background total reference attention: `0.2214`.

## Decision

- conclusion: **LEAK-FOUND**
- reasons: `['decoded reference changes far outside mask (1.000 > 0.020)', 'background queries allocate too much mass to reference segments (0.221 > 0.150)']`
- remediation order when leakage is present: erode the reference mask by 2-3 pixels and rebuild only reference keys; then adjust position offsets; finally calibrate neutral gray to the training-background statistic.
- decoded panels: `*_reference_decode.png`; manually verify garment limbs/head contamination and hair-only support before accepting the automatic conclusion.
