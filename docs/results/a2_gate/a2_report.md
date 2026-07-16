# A2 Differential Evaluation Report

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
DeltaID mean=0.4201, median=0.4285
sim_target mean=0.4388; sim_source mean=0.0186

| face_size bucket | count | mean | median |
|---|---:|---:|---:|
| 80-120 | 85 | 0.3379 | 0.3375 |
| >120 | 315 | 0.4423 | 0.4602 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/deltaid_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/deltaid_hist.png`

## Official GarmentSim

Feature: DINOv2 patch tokens pooled only over `cloth_safe` mask patches; mask outside set to gray.
cross-identity per-mid DINO mean=0.9016, median=0.9086
pairwise DINO mean=0.9016, median=0.9166, pairs=1400
generated-vs-source DINO mean=0.8358, median=0.8524
secondary cloth LPIPS mean=0.1688, median=0.1470
secondary cloth SSIM mean=0.8205, median=0.8561
CSV pairwise: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/garment_pairwise_dino.csv`
CSV source: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/garment_source_dino.csv`

## Head Pose MAE

valid images: 400 / 400; failed: 0
yaw MAE mean=16.8506, median=12.2341
pitch MAE mean=8.7316, median=7.8033
roll MAE mean=11.7279, median=9.8948

| target yaw bucket | count | yaw MAE mean | yaw MAE median |
|---|---:|---:|---:|
| front | 296 | 13.8572 | 10.8982 |
| side | 16 | 47.2305 | 50.1040 |
| three-quarter | 88 | 21.3957 | 19.2597 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/headpose_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a2_metrics/headpose_yaw_abs_err_hist.png`

## A2 Readout

ΔID 明显 > 0: adapter 起效，B2 成立，以下数字为 A2 对照基线.
A2 identity acceptance PASS: sim_target=0.4388 >= 0.300 and DeltaID mean=0.4201 > 0.
GarmentSim leaves measurable room for A2 comparison.
GarmentSim vs previous dead-adapter B2: current=0.9016, previous=0.9240, delta=-0.0224. The expected drop exposes real room for A2 differential improvement.
Head pose MAE: yaw=16.8506, pitch=8.7316, roll=11.7279
