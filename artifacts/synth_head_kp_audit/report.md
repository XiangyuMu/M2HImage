# Synthetic Head-Keypoint Audit

Conclusion: **SYSTEMATIC-BIAS**.

Current template median normalized point error: `0.023413`.
Grid-fitted median error: `0.019248` (17.8% relative improvement).
Current shoulder/nose scales: `0.480` / `1.620`; fitted: `0.470` / `1.550`.
Median center bias: dx=`-4.53px`, dy=`-4.18px`.

## Synthetic-Control History

- `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_region_resume_r16_4400_768x1024/logs/train.jsonl`: rows=462, mean ratio=`0.487012987012987`, near step-2000=[{'step': 2000, 'ratio': 0.8333333333333334}, {'step': 2002, 'ratio': 0.3333333333333333}, {'step': 2005, 'ratio': 0.8333333333333334}, {'step': 2008, 'ratio': 0.3333333333333333}, {'step': 2010, 'ratio': 0.6666666666666666}, {'step': 2011, 'ratio': 0.6666666666666666}, {'step': 2014, 'ratio': 0.5}, {'step': 2020, 'ratio': 0.0}]
- `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_incontext_resume_v2_r16_4400_768x1024/logs/train.jsonl`: rows=234, mean ratio=`0.49002849002849`, near step-2000=[{'step': 2039, 'ratio': 0.3333333333333333}, {'step': 2040, 'ratio': 0.5}, {'step': 2041, 'ratio': 0.6666666666666666}, {'step': 2043, 'ratio': 0.3333333333333333}, {'step': 2045, 'ratio': 0.5}, {'step': 2047, 'ratio': 0.3333333333333333}, {'step': 2049, 'ratio': 0.5}, {'step': 2050, 'ratio': 0.6666666666666666}]

## Visual Overlays

Green is the real mannequin DWPose head; red is the synthesized theta template.
- [00035](00035_overlay.png)
- [00430](00430_overlay.png)
- [00006](00006_overlay.png)
- [00093](00093_overlay.png)
- [00041](00041_overlay.png)
- [00807](00807_overlay.png)
- [00019](00019_overlay.png)
- [00654](00654_overlay.png)
- [00571](00571_overlay.png)
- [02331](02331_overlay.png)
- [00096](00096_overlay.png)
- [00970](00970_overlay.png)
- [00621](00621_overlay.png)
- [02811](02811_overlay.png)
- [00126](00126_overlay.png)
- [01085](01085_overlay.png)
- [00664](00664_overlay.png)
- [03803](03803_overlay.png)
- [00169](00169_overlay.png)
- [01785](01785_overlay.png)
