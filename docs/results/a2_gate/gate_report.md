# A2 vs B2-cont Gate Report

**FAIL: 差分无增量——停止堆 loss，回到方案层重估差分的价值锚点（姿势方差/弱身份区/更难身份对），先出诊断再决定 A3/A4 是否继续**

The decision comparison is A2 vs B2-cont with equal continuation steps. B2-prime is reference only.

## Fairness

fairness status: PASS
details: {'a2_run': '/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase2_a2_diff_r16_4000_768x1024', 'b2cont_run': '/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase2_b2_cont_r16_4000_768x1024', 'fields': {'resume_trainable_hash': ('c4638ff746e792e9', 'c4638ff746e792e9'), 'resume_sampler_state': ({'epoch': 0}, {'epoch': 0}), 'train_ids_hash': ('45c2b3d6ee738f81', '45c2b3d6ee738f81'), 'train_sample_count': (36033, 36033), 'excluded_train_ids': (['47160'], ['47160']), 'seed': (20260706, 20260706), 'global_batch': (18, 18), 'effective_lr': (5.6250000000000005e-05, 5.6250000000000005e-05), 'continuation_steps': (4000, 4000), 'lora_rank': (16, 16), 'lr_schedule': ('constant', 'constant')}, 'mismatches': {}, 'comparison': 'A2 vs B2-cont is the decision comparison; B2-prime is reference only.'}

## Three-metric Comparison

| metric | A2 | B2-cont | B2-prime reference |
|---|---:|---:|---:|
| DeltaID mean | 0.4201 | 0.4088 | 0.4063 |
| held-out sim_target mean | 0.4388 | 0.4223 | 0.4196 |
| GarmentSim per-mid mean | 0.9016 | 0.8951 | 0.8622 |
| GarmentSim bottom-quartile mean | 0.8469 | 0.8406 | 0.8073 |
| head-pose yaw MAE | 16.8506 | 16.9841 | 17.0474 |
| head-pose pitch MAE | 8.7316 | 8.6243 | 8.6962 |
| head-pose roll MAE | 11.7279 | 11.6718 | 11.2854 |
| pose cross-ID mean-axis variance | 89.0563 | 90.5637 | 89.3842 |

## GarmentSim Tail Analysis

histogram: /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/gate_garment_per_mid_hist.png

| run | P10 | P25 | P50 | P75 | bottom-quartile mean |
|---|---:|---:|---:|---:|---:|
| A2 | 0.8604 | 0.8835 | 0.9086 | 0.9300 | 0.8469 |
| B2-cont | 0.8445 | 0.8794 | 0.9010 | 0.9208 | 0.8406 |
| B2' ref | 0.8070 | 0.8314 | 0.8640 | 0.8967 | 0.8073 |

paired mids: 50
A2-B2-cont mean gain: 0.0065
A2-B2-cont bottom-quartile gain: 0.0063
Wilcoxon one-sided (A2 > B2-cont): p=0.066480, rank-biserial effect=0.2455

## Pose Cross-identity Variance

paired mids: 50
A2-B2-cont mean variance delta: -1.5074 (negative is better)
Wilcoxon one-sided (A2 < B2-cont): p=0.053688, rank-biserial improvement effect=0.2627

## Identity Preservation

paired images: 400
A2-B2-cont DeltaID mean delta: 0.0113
Wilcoxon regression test (A2 < B2-cont): p=0.999997
identity regression: NO

## Automatic Rule Inputs

significant mean GarmentSim gain: False
bottom-quartile gain >= 0.02: False
DeltaID significant regression: False
