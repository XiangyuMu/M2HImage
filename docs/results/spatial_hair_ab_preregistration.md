# Spatial Hair A/B and New-System Directed Identity Preregistration

Frozen before formal training: 2026-08-25 11:33:54 CST.

## Part A diagnosis

- Verdict: CAPACITY-OR-TRAINING.
- sleeve_mismatch: 35.5% (142/400).
- sim_target P25/P75: 0.30149 / 0.49517.
- Hair labels do not satisfy the 2x low-vs-high enrichment rule.
- The semantic hair A/B remains required, while Part C retains high priority.

## Frozen inputs

- Resume checkpoint SHA256: a1219cba4946e4068565394485399076e8a4ef1b623e2fe4c509cbc48fac0e05.
- Train split SHA256: 38d1d8b58bf279aeed38a66525f422b5540eff2deca3833603cac2532d5e1615.
- cf_subset SHA256: 7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985.
- identity_bank_v2 SHA256: 1772598863bca76b2d743cca79fcb862155d949cc848fd330f01ae83bc0e4e9d.
- Cache coverage: 40,014 / 40,014, zero shard failures.
- Seed: 20260706.

## Part B fixed gate

B-space and B-semantic both resume step 6400 and execute exactly 2000
optimizer steps with the same seed, sampler state, train IDs, LR, rank, and
global batch.

B-semantic passes only if all conditions hold:

1. Hair-DINO >= B-space - 0.02.
2. face_occluded_by_hair is lower and paired McNemar/binomial p < 0.05.
3. sim_target >= B-space + 0.02.
4. Garment-DINO >= B-space - 0.005.
5. head5 <= B-space + 0.002.

SEMANTIC-HAIR-FAILED is emitted when the 2000-step hair_gate movement is below
0.01 and Hair-DINO is below 0.60. Any non-pass that does not meet that special
failure rule selects B-space without a rescue run.

## Part C fixed gate

From the Part B winner, C-A4prime and C-cont execute exactly 4000 steps.

PASS requires all conditions:

1. held-out sim_target gain >= 0.03 and paired greater-side Wilcoxon p < 0.05.
2. Garment-DINO >= C-cont - 0.005.
3. head5 <= C-cont + 0.002.
4. face detection rate >= 99%.
5. Print-HF-LPIPS does not increase and its paired greater-side degradation
   test is not significant.

The report always compares old-system identity gain / GarmentSim cost against
the new-system values. No post-hoc third mechanism run is admitted.

## Preflight evidence

- 25 focused tests passed.
- B-space smoke: 20/20 steps, 43.655 GiB, 20.58 s/step, hair skip 10%.
- B-semantic smoke: 20/20 steps, 42.434 GiB, 18.25 s/step.
- Semantic hair_gate: 0.100132 -> 0.114526 in 20 smoke steps.
- Fixed-set step-6400 medians: Garment-DINO 0.94979, Hair-DINO 0.81412,
  head5 0.01734; face detection 100%.
