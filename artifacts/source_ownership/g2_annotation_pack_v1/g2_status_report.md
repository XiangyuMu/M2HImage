# G2 status — source ownership ontology / hair policy

Date: 2026-09-05 (Asia/Shanghai)

## Decision

**G2 is BLOCKED/PENDING, not PASS.** The protocol machinery is implemented and its synthetic tests pass, but the empirical hair-policy gate cannot be evaluated because the three independent annotation sheets are still blank. No 2×2 generation, G4 image evaluation, full-model training, or new-method training was started.

## Frozen protocol artifacts

- `source_ownership_ontology.py`
- `configs/source_ownership_ontology_v1.yaml`
- `docs/source_ownership_annotation_protocol_v1.md`
- `tools/build_g2_annotation_pack.py`
- `tools/evaluate_g2_hair_policy.py`
- `tests/test_source_ownership_ontology.py`
- `artifacts/source_ownership/g2_annotation_pack_v1/`
- `complete_2x2_blocks.py`
- `directional_metrics.py`
- `injected_metric_validation.py`
- `tests/test_g3g4_protocol.py`

The pack contains 30 validation samples, balanced 15/15 across `zalando_M2H` and `VITON-HD_M2H`, with 30 distinct near-duplicate clusters. It stores paths and SHA-256 hashes only; it does not copy source images.

## Required empirical evidence still missing

Each of the following must be completed independently for all 30 images and all 210 units per rater:

1. `annotations_rater_01.csv`
2. `annotations_rater_02.csv`
3. `annotations_rater_03.csv`
4. blinded `adjudication.csv`

The evaluator rejects empty support labels, owner-observability fields, polygons, occlusion order, and adjudicated positive area. Therefore a blank pack cannot produce a false PASS. The evaluator also requires a separately frozen face-only policy artifact before a failed revised hair gate can count as a valid fallback PASS.

## Code evidence

- Remote interpreter: `/home/muxiangyu/miniconda3/bin/python`
- `test_source_ownership_ontology.py`: 9/9 tests PASS.
- `test_g3g4_protocol.py`: 8/8 tests PASS after fixing coordinate-wise swap validation.
- `py_compile` passed for all new Python modules.
- Deterministic pack replay: PASS; source hashes: PASS; source balance: PASS; distinct near-duplicate clusters: PASS.
- Running the evaluator on the blank pack fails closed with `ProtocolError: missing support_label` and creates no gate report.
- Directional/2×2 code is synthetic-test-only at this stage; no generated images or real metric-injection run exists.
- Remote `/data` reported 399 GB available at the latest audit; generated Python caches were removed after testing.
- A conditional G3 engineering manifest is now prepared at remote `artifacts/source_ownership/g3_pilot_manifest_20.csv`: 20 blocks/80 cells, 10 blocks per synthetic source, 80 unique source images and near-duplicate clusters, exact same-noise hashes, and deterministic replay SHA-256 `d7655c4ffa14c7fe66b3286fe9cff80e07d30cc601c4d6d2fe480c04d363928f`. It is marked `prepared_not_run`; no generated images exist under `artifacts/source_ownership`.

## Gate rule

Per the approved PRD/test spec, continue with the main-hair policy only if alpha ≥ 0.80, median boundary IoU ≥ 0.75, occlusion kappa ≥ 0.70, and at least 90% of images have adjudicated `S_U` ≤ 10%. After one failed revision, use face-only only when that policy is explicitly frozen and verified. Until then, G2 remains open and downstream gates may not advance.
