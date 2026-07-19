# A4 Directed Counterfactual Offline Metric Report

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
DeltaID mean=0.4794, median=0.5028
sim_target mean=0.4944; sim_source mean=0.0150

| face_size bucket | count | mean | median |
|---|---:|---:|---:|
| 80-120 | 84 | 0.4274 | 0.4321 |
| <80 | 1 | 0.2827 | 0.2827 |
| >120 | 315 | 0.4939 | 0.5198 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/deltaid_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/deltaid_hist.png`

## Official GarmentSim

Feature: DINOv2 patch tokens pooled only over `cloth_safe` mask patches; mask outside set to gray.
cross-identity per-mid DINO mean=0.8778, median=0.8809
pairwise DINO mean=0.8778, median=0.8883, pairs=1400
generated-vs-source DINO mean=0.7828, median=0.7807
secondary cloth LPIPS mean=0.2207, median=0.2265
secondary cloth SSIM mean=0.7522, median=0.7593
CSV pairwise: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/garment_pairwise_dino.csv`
CSV source: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/garment_source_dino.csv`

## Head Pose MAE

valid images: 400 / 400; failed: 0
yaw MAE mean=16.9204, median=12.7963
pitch MAE mean=8.7353, median=7.8443
roll MAE mean=11.3662, median=9.5638

| target yaw bucket | count | yaw MAE mean | yaw MAE median |
|---|---:|---:|---:|
| front | 296 | 13.8585 | 11.0785 |
| side | 16 | 48.3825 | 50.9041 |
| three-quarter | 88 | 21.4991 | 19.9686 |

CSV: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/headpose_per_image.csv`
Histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_metrics/headpose_yaw_abs_err_hist.png`

## A4 Readout

These frozen metrics are inputs to `eval_a4_gate_report.py`; A4 has no standalone verdict against B2-prime.
ΔID 明显 > 0: adapter 起效，B2 成立，以下数字为 A2 对照基线.
GarmentSim leaves measurable room for A2 comparison.
Head pose MAE: yaw=16.9204, pitch=8.7353, roll=11.3662
