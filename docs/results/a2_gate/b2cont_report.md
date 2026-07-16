# B2-cont Equal-step Evaluation Report

subset: 50 mannequins / 20 identity pool / 200 pairs
garment type counts: {'dress': 13, 'pants': 13, 'skirt': 12, 'top': 12}
expected generated images: 400
actual generated images: 400
generation status: complete

## Metric Provenance

Held-out identity recognizer: UniFace AdaFace IR-101 ONNX | status=ok | hash=f2eb07d03de0
Face detection/alignment for DeltaID uses InsightFace RetinaFace only; recognition uses the held-out recognizer when configured.
DINO garment encoder: GSVTON vendored DINOv2 ViT-B/14 | status=ok | hash=0b8b82f85de9
Head pose runner: uniface_headpose | status=ok | hash=61c34e877989
Mask projection: source mid derived/region_masks/{mid}.npz cloth_safe resized to configured generated frame; mask outside set to gray before DINO

## Held-out DeltaID

valid images: 400 / 400; failed: 0
DeltaID mean=0.4088, median=0.4268
sim_target mean=0.4223; sim_source mean=0.0135

| face_size bucket | count | mean | median |
|---|---:|---:|---:|
| 80-120 | 86 | 0.3438 | 0.3444 |
| <80 | 1 | 0.2255 | 0.2255 |
| >120 | 313 | 0.4273 | 0.4406 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/deltaid_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/deltaid_hist.png`

## Official GarmentSim

Feature: DINOv2 patch tokens pooled only over `cloth_safe` mask patches; mask outside set to gray.
cross-identity per-mid DINO mean=0.8951, median=0.9010
pairwise DINO mean=0.8951, median=0.9060, pairs=1400
generated-vs-source DINO mean=0.8207, median=0.8337
secondary cloth LPIPS mean=0.1855, median=0.1750
secondary cloth SSIM mean=0.7926, median=0.8126
CSV pairwise: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/garment_pairwise_dino.csv`
CSV source: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/garment_source_dino.csv`

## Head Pose MAE

valid images: 400 / 400; failed: 0
yaw MAE mean=16.9841, median=12.3509
pitch MAE mean=8.6243, median=7.6733
roll MAE mean=11.6718, median=9.9645

| target yaw bucket | count | yaw MAE mean | yaw MAE median |
|---|---:|---:|---:|
| front | 296 | 13.9803 | 10.9618 |
| side | 16 | 47.0123 | 50.2626 |
| three-quarter | 88 | 21.6281 | 19.3481 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/headpose_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/b2cont_metrics/headpose_yaw_abs_err_hist.png`

## B2-cont Readout

ΔID 明显 > 0: adapter 起效，B2 成立，以下数字为 A2 对照基线.
B2-cont identity acceptance PASS: sim_target=0.4223 >= 0.300 and DeltaID mean=0.4088 > 0.
GarmentSim leaves measurable room for A2 comparison.
GarmentSim vs previous dead-adapter B2: current=0.8951, previous=0.9240, delta=-0.0289. The expected drop exposes real room for A2 differential improvement.
Head pose MAE: yaw=16.9841, pitch=8.6243, roll=11.6718
