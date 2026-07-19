# A4 One-shot Identity-axis Gate Report

**MIXED: 身份-服装 trade-off 成立，作为发现如实报告，训练侧不再重跑，λ 权衡留给论文讨论**

This is the preregistered final mechanism run. The decision comparison is A4 vs the existing equal-step B2-cont; A2 is reference only. No third rescue run is allowed.

## Fairness

fairness: PASS
details: {'fields': {'resume_trainable_hash': ('c4638ff746e792e9', 'c4638ff746e792e9'), 'resume_sampler_state': ({'epoch': 0}, {'epoch': 0}), 'train_ids_hash': ('45c2b3d6ee738f81', '45c2b3d6ee738f81'), 'train_sample_count': (36033, 36033), 'excluded_train_ids': (['47160'], ['47160']), 'seed': (20260706, 20260706), 'global_batch': (18, 18), 'effective_lr': (5.6250000000000005e-05, 5.6250000000000005e-05), 'continuation_steps': (4000, 4000), 'lora_rank': (16, 16), 'lr_schedule': ('constant', 'constant')}, 'mismatches': {}}

## Three-run Comparison

| metric | A4 | B2-cont | A2 reference |
|---|---:|---:|---:|
| held-out sim_target mean | 0.4944 | 0.4223 | 0.4388 |
| DeltaID mean | 0.4794 | 0.4088 | 0.4201 |
| GarmentSim per-mid mean | 0.8778 | 0.8951 | 0.9016 |
| pose cross-ID mean-axis variance | 88.7023 | 90.5637 | 89.0563 |
| face detection rate | 1.0000 | 1.0000 | 1.0000 |
| face detector confidence mean | 0.8434 | 0.8409 | 0.8428 |
| head-pose yaw MAE | 16.9204 | 16.9841 | 16.8506 |
| head-pose pitch MAE | 8.7353 | 8.6243 | 8.7316 |
| head-pose roll MAE | 11.3662 | 11.6718 | 11.7279 |

## Identity Primary Gate

- paired images: 400
- sim_target mean gain A4-B2-cont: 0.072096 (required >=0.03)
- Wilcoxon greater p=0.00000000, rank-biserial=0.9135
- DeltaID mean gain: 0.070592, greater p=0.00000000
- identity condition: PASS

## Treatment Strength

- histogram: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/a4_treatment_strength.png`
- semi-hard relaxation counts: {'strict': 72000, 'relax_age': 0, 'relax_skin': 0}
- strengthened in F_train space: YES
- The random baseline replays the exact A2 random pairing policy but measures both policies in bank-v2 F_train space; old-bank and bank-v2 cosine distances are not mixed.

| policy | count | mean | P25 | P50 | P75 |
|---|---:|---:|---:|---:|---:|
| A2 random policy replay, bank v2 | 72000 | 0.9420 | 0.8932 | 0.9503 | 1.0036 |
| A4 semi-hard actual | 72000 | 1.1000 | 1.0736 | 1.0977 | 1.1237 |

training sim_gap/skip summary: {'triggered_log_rows': 956, 'attempts': 1387.0, 'skips': 27.0, 'skip_rate': 0.01946647440519106, 'sim_gap_first_quartile_mean': 0.07871480214356276, 'sim_gap_last_quartile_mean': 0.15987744788292438, 'sim_gap_latest': 0.03261987864971161}

## Non-regression Gates

| constraint | A4 | B2-cont | one-sided deterioration p | result |
|---|---:|---:|---:|---|
| GarmentSim per-mid mean | 0.8778 | 0.8951 | 0.00102455 | REGRESSION |
| pose cross-ID variance | 88.7023 | 90.5637 | 0.89948263 | PASS |
| face detector confidence | 0.8434 | 0.8409 | 0.99807992 | PASS |
| face detection rate | 100.00% | 100.00% | N/A | PASS |

Face detector confidence is the automatic face-realism proxy available in the frozen shared runner; recognition remains held-out AdaFace IR-101.

## Automatic Rule Inputs

- identity primary condition: True
- garment non-regression: False
- pose non-regression: True
- face detection/realism non-regression: True
- treatment strength increased: True
