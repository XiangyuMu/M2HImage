# M2H layered data protocol v1

Status (2026-09-07 execution): exploratory dev128 v2 FROZEN and validated;
candidate grouped manifests emitted; formal clean split OPEN.

## Executed freeze and correction

`data_protocol_v1` was rejected before A generation because its reuse-first
allocator yielded only3 reference files. The v2 allocator fills at least32
reference files before reusing; thresholds, M pool, seed and image-risk checks
are unchanged. Deployed filename `scripts_controls_v1/freeze_data_protocol_v2.py`.
`data_protocol_v2/dev128_pairs.json` has64 unique source groups,32 reference files,
128 pairs, each reference used exactly4 times. Maximum selected M/ref raw Glint
cosine .2803753636. Maximum reference/reference cosine across all32 files .777191;
zero pairs >=.9. These are embedding checks, not biological identity ground truth.
Manifest SHA256 `9cc9bc517e65486c2794cde0ef1d002d8db0f3f186bc2adc0c8fcdc5843929be`.
The minimum-reference regression test and seven prior selector tests passed.

Group audit and non-destructive candidates are documented in
`DATA_PROTOCOL_CLOSEOUT.md`: corrected joint graph19,303 components, max282;
candidate train33302/dev1845/final1833/quarantine3034. Original metadata/splits
unchanged. No formal-ready flag, no clean model training, no actual final test.
Candidate final metadata is visible for audit; no final evaluation outcomes
were generated. Historical checkpoint contamination persists after re-splitting.

The source-M boundary proxy calibration failed to establish reliable real-defect
ordering; no artificial pass threshold or human label was added to rescue it.
Current diagnostics can proceed, while formal P0 promotion remains withheld.

## Original interface rationale
Owner edits only this document, `scripts/freeze_data_protocol.py`, and its test.
Main agent owns remote execution, A/B/cache code and STATUS.md.

## Immediate A interface

`dev128_pairs.json` is an object with `pairs`, exactly 128 rows:

```json
{"pairs":[{"mid":"00001","jid":"00002","seed":0,"donor_jid":"00003"},{"mid":"00001","jid":"00003","seed":0,"donor_jid":"00002"}]}
```

Example IDs illustrate schema only. Preserve string IDs/leading zeroes.
Actual selection preserves the existing ordered 64 IDs in
`p0_metadata_20260907/old_val_probe_ids.json`. Each M has exactly two different
old-val reference IDs, both outside the entire M pool. `donor_jid` is the other
reference of this M, so matched/permuted arms use the same teacher endpoint
multiset with a no-fixed-point swap. Generation seed is 0 for every row.
Cache builder consumes `--pairs-json <out>/dev128_pairs.json`.
Teacher generation failures remain in the denominator; no quality screening.

Selection is bounded to the existing old-val IDs and at most 128 reference IDs.
It reuses eligible pool members with usage balancing before adding references;
it does not promise 32 biological identities or 128 unique reference files.
Selection ordering uses SHA256 with fixed seed 20260907. A failed allocation
raises an error; it never silently substitutes M IDs or relaxes thresholds.

Direct float64 cosine of normalized existing `embeddings_*.npz` is required.
Every M/ref and that M's ref/ref pair has raw Glint cosine **strictly < 0.3**.
This is a preregistered high-confidence-not-same *embedding proxy*, not a
calibrated probability or a verified biological identity label. It compares
the associated human H_i with H_j; H_i is permitted for dataset selection only,
not for input cache/inference. No top16 omission is used as negative evidence.
No AdaFace invocation, inference, GPU or dependency installation is involved.

All refs exclude every M source-parent, exact SHA256 overlap and dHash Hamming
distance <=4 against either audited image role. Different reference files also
exclude those source/hash risks. These conservative image proxies can reject
legitimate images and miss semantic/near duplicates. No claim of exhaustive
perceptual duplicate clearance follows. M IDs are retained even if future
audits reveal dependencies among them; report effective source count and do
not treat 128 pairs as independent observations.

## Source semantics and evidence

Read-only inspection 2026-09-07 confirmed 40,014 manifest IDs (28,950 Zalando,
11,064 VITON-HD), 80,028 image audit records with zero failures, 111 exact
groups and 20 cross-original-split exact groups. Existing full features cover
all 40,014 human images. Glint SHA256:
`4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf`.
The loader verifies metadata digest, bank coverage, finite unit vectors,
feature-to-human-audit hashes and original split membership, and records hashes
of every input shard/audit plus its own script. It does not rehash 80k images;
audited image immutability is an explicit execution assumption.

Observed grammar: `multiimages_x2__00001_1/2` are view-index candidates;
`singleimage_x2__10158` identifies a single item; `train__14684_00` and
`test__00006_00` retain original VITON namespace and filename. Only the explicit
multiimages grammar loses its last view index. Other keys remain opaque.
Old generic `rpartition('_')` strips the item number of every singleimage key
and incorrectly combines that namespace. Existing giant-component statistics
therefore mix a source-parser artifact with real embedding connectivity.
The original exporter/SKU ontology is not available in the inspected repo;
source groups remain provenance candidates, not established garment/identity
truth. Keep old audits untouched; do not relabel their counts as corrected.

## Roles and promotion boundaries

- Exploratory dev128: frozen inputs for A-D0/B controls, proxy calibration and
  method selection. Historical space8400 can be used here, with contamination
  explicitly disclosed. Dev teachers are diagnostic caches, never training
  examples for a result evaluated on this same dev set.
- Training: formal future train manifest only for optimization, task adapters,
  teacher/EMA pools and all cache learning. Exploration continued from the old
  model remains exploratory even if new examples happen to avoid holdout IDs.
- Development: future dev manifest only for thresholds, checkpoint and method
  selection. Independent frozen buffalo_l recognizer is the established dev
  candidate in STATUS; its upstream provenance remains a publication gap.
- Final: separate unopened manifest and evaluation configuration, frozen hashes,
  no training/teacher/selection access. Final-only AdaFace is forbidden here.
  Historical 400 and old test are not new final tests. Real-domain authorized
  data availability and power/size checks remain OPEN.
- Formal clean: freeze defensible train/dev/final membership and exclusions,
  then retrain all task-adapted parameters from common public pretrained
  initialization. Old checkpoint contamination cannot be fixed by a manifest.
  Public-pretraining overlap remains a disclosed limitation.

## Main-agent execution (CPU only)

After the main agent stages the owned script into its remote artifact scripts
directory, run in P. No remote staging or execution is performed by this owner.

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python artifacts/m2h_ab_objective_v1/scripts/freeze_data_protocol.py --root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2 --artifacts /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1 --out /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/data_protocol_v1
```

Use a fresh output directory for a new version; reruns cannot overwrite.
Success writes `dev128_pairs.json`, `summary.json`, `DEV128_READY` (manifest
SHA256). No original split or image is edited or deleted. `summary.json` exposes
separate exploratory readiness and formal/clean-retraining false flags for
STATUS.md consumption. The marker proves selection completion only; main
agent must separately verify input-only caches and inference behavior.

Run tests with the existing NumPy environment:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -m unittest discover -s artifacts/m2h_ab_objective_v1/scripts -p test_freeze_data_protocol.py -v
```
