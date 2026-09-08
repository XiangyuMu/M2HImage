# Data protocol grouping audit closeout

2026-09-07, independent CPU audit. **Audit complete; candidate manifests emitted;
formal split BLOCKED.** Biological identity and source ontology calibration remain
gaps. This closes the grouping evidence task, not the formal data protocol.

## Scope and owned paths

Only these two local files were added under this experiment:

- `scripts/protocol_group_audit.py`
- `DATA_PROTOCOL_CLOSEOUT.md` (this report)

Read the existing `scripts/freeze_data_protocol.py`, `scripts/group_sensitivity.py`
and `DATA_PROTOCOL.md`; did not edit them. No dev128 generation, A/B work, GPU
inference, training, manual labels, threshold search, dataset edits or original
split edits. Existing feature/neighbor artifacts were produced earlier using GPU;
the new audit itself uses only NumPy on CPU, with GPU visibility disabled.

Remote constants:

```text
Host = 10.249.45.227
P = /data/muxiangyu/pythonPrograms/M2HImage
D = /data/muxiangyu/datasets/M2HImage/M2H_Final_v2
R = /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1
Python = /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
New audit directory = R/protocol_group_audit_v1
Evidence directory = R/protocol_group_audit_v1/results
```

All new remote writes were confined to that new audit directory. SSH used
`-o BatchMode=yes -o ClearAllForwardings=yes`. The repository workspace is not a
Git checkout (`git status` reports no repository); ownership was enforced by
targeting only the two authorized local paths.

## Corrected source semantics

Only `^(multiimages_x2__\d+)_\d+$` loses its final view index. Thus
`multiimages_x2__00001_1` and `_2` share a provenance candidate parent, while
`singleimage_x2__10158` retains the complete item key. Unknown patterns and VITON
`train__14684_00` / `test__00006_00` stay opaque, including namespace and suffix.
The rule matches the existing freeze script; the old sensitivity script remains
untouched and its results remain labeled legacy.

The old generic `rpartition('_')` merged **8,387 singleimage records into one
source group**. The corrected audit gives those records separate item keys.
There are 19,451 changed key strings: 8,387 singleimage plus 11,064 VITON keys.
The VITON key spelling restoration does not change source-group membership in
this corpus; the net source component increase is 8,386. These are provenance
proxies, not validated garment/SKU or biological identity labels.

## Measured groups and original-split crossings

All counts below use the same 40,014 string IDs, preserving leading zeroes.
Original splits: train 36,034; val 1,996; test 1,984. Components include singletons;
cross-split record counts count each record once within the stated stage.

| Graph stage | Components | Largest | Cross-original-split components | Records in crossing components |
| --- | ---: | ---: | ---: | ---: |
| Legacy source only | 22,777 | 8,387 | 1,571 | 11,551 |
| Corrected source only | 31,163 | 4 | 1,570 | 3,164 |
| Exact SHA256 only, both image roles | 39,903 | 2 | 20 | 40 |
| Identity top16 cosine ≥0.9 only | 27,087 | 259 | 1,088 | 8,867 |
| Corrected source + exact | 31,053 | 4 | 1,590 | 3,204 |
| Corrected source + exact + identity | 19,303 | 282 | 1,961 | 13,389 |
| Legacy source + identity, historical reproduction | 12,911 | 17,036 | 1,381 | 21,673 |

The historical `group_sensitivity/summary.json` threshold-0.9 component count and
largest component were independently reproduced exactly. The legacy 17,036-record
giant therefore must not be reported as a corrected-source finding. At the fixed
threshold used here, the corrected joint graph has no component larger than the
original smaller holdout budget of 1,984. This finding does not cover other
thresholds or the unenumerated full threshold graph.

Corrected source size distribution: 22,400 singletons, 8,677 pairs, 84 triples,
2 groups of four. Corrected joint graph: 11,510 singletons and 7,793 non-singleton
components. Full size histograms and crossing patterns are in `summary.json`;
`components.jsonl` includes every component's members, original-split counts and
source-dataset counts for all seven stages.

All-role exact SHA256 grouping reconstructed from 80,028 successful image-audit
records yields 111 duplicate groups, each spanning two IDs; 20 cross original
splits (12 train/val, 8 train/test). This agrees with the historical role-specific
summary; equality was checked across both human and mannequin roles in this run.

The fixed identity graph has 55,063 directed cached edges, 28,355 unique undirected
edges, and 9,886 directed edges crossing original splits. **356 rows have their
16th neighbor at or above 0.9**, so the cache can omit other qualifying edges.
Every cached top16 cosine was checked against the normalized full Glint feature
bank on CPU: maximum absolute difference `1.3403400767053597e-06`. There are 48
cached directed entries within `1e-5` of 0.9; the graph uses the original cached
values and the fixed ≥0.9 rule, without precision-based threshold adjustment.
This validates cached edge values, not completeness or optimality of top16.

## Candidate manifests and quarantine

Policy fixed in the script before running:

1. Union corrected source links, all-role exact hashes and cached identity edges
   at top16 cosine ≥0.9; treat each entire component as indivisible.
2. Quarantine every component larger than the original smaller holdout (1,984
   records), or containing any top16-saturated row. This is an engineering
   exclusion rule; it is not identity calibration or outcome screening.
3. Assign remaining components largest first, using SHA256 with seed 20260907 for
   equal-size ordering and the largest remaining target deficit for allocation.
   Train/dev/final fractions inherit original train/val/test proportions. No
   model result, manual annotation or threshold trial enters assignment.

| Candidate partition | Records |
| --- | ---: |
| train | 33,302 |
| dev | 1,845 |
| final | 1,833 |
| quarantine | 3,034 |
| Total | 40,014 |

Retained: 36,980 records in 19,267 components. Quarantine: 3,034 records in 36
components, all due to top16 saturation; oversize components: zero. Quarantine
contains 2,723 original-train, 142 original-val and 169 original-test records;
2,343 Zalando and 691 VITON-HD records. Exclusion is a manifest designation only:
no image was moved or deleted. Retained counts are within one record of the
proportional targets. Quarantine removes about 7.58% of the corpus and changes the
population; no representativeness or statistical-power claim follows.

All 64,025 serialized evidence edges (8,851 source spanning edges, 111 exact
spanning edges, 55,063 directed identity edges) have **zero candidate-partition
crossings**, including quarantine boundaries. Original split columns and image
paths remain in each manifest for traceability. Candidate `final` is just a
proposed partition label; it is not an unopened, approved final evaluation set.

## Artifacts

All following paths are relative to remote `R/protocol_group_audit_v1/results/`:

- `summary.json`: fixed policy, stage statistics, measured gaps and false formal
  readiness flags; status `DIAGNOSTIC_ONLY_FORMAL_BLOCKED`.
- `components.jsonl`: stable SHA256 component IDs and complete membership for
  each graph stage, with component sizes and original-split/source breakdowns.
- `edges.csv`: source/exact spanning witnesses and directed cached identity
  edges with raw cosine values. Source/exact equivalence classes need only
  spanning links to establish the same components.
- `source_key_corrections.csv`: all 19,451 old/new source key mappings.
- `candidate_all.csv`: full disjoint assignment, including quarantine.
- `candidate_train.csv`, `candidate_dev.csv`, `candidate_final.csv`: proposed
  retained partitions; each row has `formal_split_ready=false`.
- `quarantine.csv`: excluded component membership and explicit reason.
- `provenance.json`: SHA256 digests of original metadata/splits, audited inputs,
  every feature shard, neighbor cache and deployed script.
- `artifact_sha256.json`: digests of every result file preceding this index.
- `AUDIT_COMPLETE_NOT_FORMAL_READY`: summary digest; confirms completed audit
  only. There is no formal-ready marker.

Remote result files total 74,438,027 bytes. Key SHA256 values:

```text
protocol_group_audit.py b154a01638e02f850c63b402549d3375782a034a19da3355e932d8d3e949d8a8
summary.json           4b415a35000d2bb8d21f5a8af09726676df65480b39e186a58aa3622f5cbdee4
candidate_all.csv      dc733e8405c5f3666370ea0d55d448a571c38f3aebafff12421495dc52afcbad
candidate_train.csv    d14d171b1cd93d4eca6ab37c7d9e6b3fa29945d7b64aacd6eb2cdc0b93b1296a
candidate_dev.csv      6adbf065e8983e9dca3ec41cd48d5baf84e29e3dcd63bee2ad74d14ea214461a
candidate_final.csv    ca4af6981e9222db30a2a42fba87cd9277af291d07aa06fffcc4c541788f1cfa
quarantine.csv         a66806524eb0de9ca1c3fe47f5da1a92fd64989e68e42c38de2de479fda531e2
```

## Commands and verification

Executed from the local workspace; the remote directory did not previously
exist. Do not rerun deployment into this existing completed directory.

```sh
python3 -B .omx/experiments/m2h_ab_objective_v1/scripts/protocol_group_audit.py --self-test
ssh -o BatchMode=yes -o ClearAllForwardings=yes 10.249.45.227 'mkdir /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_group_audit_v1'
scp -o BatchMode=yes -o ClearAllForwardings=yes .omx/experiments/m2h_ab_objective_v1/scripts/protocol_group_audit.py 10.249.45.227:/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_group_audit_v1/protocol_group_audit.py
ssh -o BatchMode=yes -o ClearAllForwardings=yes 10.249.45.227 "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -B /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_group_audit_v1/protocol_group_audit.py --root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2 --artifacts /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1 --out /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_group_audit_v1/results"
```

Audit exit code 0, CPU runtime 6.081 seconds, GPU hours 0. Embedded regression
checks passed locally and remotely: distinct singleimage keys, historical bug
reproduction, multi-view grouping, opaque-key preservation, source namespaces
and transitive union. Input checks passed for all metadata/split memberships,
feature IDs and model provenance, 80,028 audited image roles, feature-to-human
hashes, full-bank normalization and cached cosine values.

A separate read-only remote Python process independently reloaded the saved
CSVs and verified: all 40,014 IDs occur exactly once; each partition CSV equals
its subset in the complete CSV; component sizes and membership SHA256 IDs match;
all evidence edges stay within a candidate partition; historical threshold-0.9
statistics reproduce; every recorded input/output digest and completion marker
matches. It returned `independent_verification: PASS`, exit code 0. Original
metadata and all three split files retained their pre-run SHA256 digests in both
the script and this independent check. No lint/typecheck tooling was installed;
the executed script, embedded regressions and independent artifact checks are
the verification evidence for this standalone addition.

## Limits and remaining formal gates

- No giant-component blockage was observed in the corrected cached graph at the
  fixed threshold. If any component exceeds the fixed holdout budget, this
  script explicitly reports `BLOCKED_OVERSIZE_COMPONENTS` and isolates it. Here
  the full-corpus holdout status is `NOT_CERTIFIED`; formal status remains
  blocked for the following unresolved reasons.
- Raw Glint cosine is not a calibrated biological identity probability;
  transitive components are not verified people. Source parent grammar is not
  an established exporter/SKU/garment ontology. No labels were invented.
- The full threshold graph was not recomputed. Missing top16 edges can merge
  cached components, and saturation quarantine does not establish exhaustive
  semantic identity separation. Float precision near the threshold is disclosed.
- Existing image-audit hashes were reused; underlying image bytes were not
  rehashed. Image immutability is an explicit assumption. Non-exact/perceptual
  duplicates and nonzero-Hamming near duplicates remain unaudited here.
- Candidate membership can move IDs across original split names. Historical
  checkpoints and teachers are therefore not made clean by these manifests;
  formal approval, clean retraining, power/coverage checks and final evaluation
  remain separate gates. Public-pretraining overlap is also unresolved.
- dev128 production and A/B remain with the main task. This side task generated
  no dev pairs, teacher caches, training runs or model comparisons.
