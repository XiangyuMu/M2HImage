# Protocol near-duplicate candidate risk audit

2026-09-07. **Radius-4 enumeration COMPLETE; formal protocol remains BLOCKED.**
These are dHash proximity candidates, not verified semantic duplicates or people.
Per the main task's subsequent direction, **the 16,732-ID conservative additional
quarantine proposal is NOT ADOPTED**. Original candidate manifests, old splits,
and frozen dev128 remain unchanged. No additional audit expansion is planned.

## Scope and execution

Only two local files were added under `.omx/experiments/m2h_ab_objective_v1/`:
`scripts/protocol_near_duplicate_audit.py` and this report. No existing local file
was edited. All remote writes are under the new directory:

```text
R = /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1
New script = R/protocol_near_duplicate_v1/protocol_near_duplicate_audit.py
Results = R/protocol_near_duplicate_v1/results/
Host = 10.249.45.227
Python = /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
```

Inputs: existing `full_duplicates/per_image.jsonl` and summary; previous
`protocol_group_audit_v1/results/candidate_all.csv`, summary and provenance; and
`identity_neighbors_full/neighbors.npz` for the conditional top16 logic check.
No model inference, training, GPU work, dependency installation, human labels or
image-byte rehashing. Python enumeration is single-threaded; NumPy/BLAS/OpenMP
thread limits are set to 2 and `CUDA_VISIBLE_DEVICES=''`.

Audit runtime: **27.854 seconds**, exit code 0, GPU hours 0. Wall-clock side-task
work started at 16:44:43 China time; enumeration and independent verification were
complete by 16:51:41, within the requested 10-minute budget. The remaining work
was documentation only.

## Exact enumeration and coverage

Fixed radius: **Hamming ≤4**, with no threshold search. Each 64-bit hash is split
from least significant bit upward into disjoint widths **13/13/13/13/12**.
If at most four bits differ, at most four of these five blocks differ; therefore
at least one block is identical. Equal-block postings consequently include every
qualifying distinct-hash pair. For each hash, earlier matching hash indices are
deduplicated in a set before XOR `bit_count()` verification. This visits each
unordered candidate at most once; no approximate nearest-neighbor search is used.

Images with equal hashes are grouped first. Each equal-hash bucket expands into
all unordered image pairs; every qualifying distinct-hash pair expands into its
full Cartesian product of image members. Cross-role pairs and human/mannequin
pairs belonging to the same ID are included. Projecting image pairs to ID edges
excludes only ID self-loops, while preserving minimum Hamming distance, number of
image witnesses, and number of exact SHA256 witnesses.

| Coverage/count | Result |
| --- | ---: |
| Successful audited image records | 80,028 |
| Dataset IDs, two image roles each | 40,014 |
| Unique 64-bit hashes | 78,822 |
| Unique hashes fully processed | 78,822 |
| Deduplicated distinct-hash candidates checked | 58,902,935 |
| Distinct-hash pairs passing radius 4 | 30,391 |
| Equal-hash buckets containing multiple images | 1,185 |
| Unordered same-hash image pairs | 1,239 |
| All passing image pairs, including same hash | 34,725 |
| Passing pairs with exact SHA256 equality | 111 |
| Passing image pairs within the same ID | 1,911 |
| Distinct ID edges | 30,604 |

The all-role equal-hash bucket count 1,185 is not the earlier role-specific dHash
candidate count 1,171; this audit pools both roles. Equality of dHash itself is
also insufficient to establish image equality: only 111 of the 34,725 passing
image pairs have identical audited SHA256 values.

Distance histogram: d=0 **1,239**; d=1 **1,285**; d=2 **2,576**; d=3 **7,521**;
d=4 **22,104**. Role histogram: human/human **22,282**; mannequin/mannequin
**7,906**; human/mannequin **4,537**.

Coverage is `true` for **all pairs of the supplied audited hashes**. No cap was
hit. Execution used a 210-second enumeration budget, 10,000,000-image-pair cap and
2,000,000-ID-edge cap. If a guard fires, the script emits `COVERAGE_INCOMPLETE`,
marks statistics as partial lower bounds and does not issue a complete marker.
This successful run fully processed and exactly verified all 58,902,935 candidates.
Completeness follows from the partition argument and complete traversal; the
independent sampled checks below are additional verification, not its sole basis.

## Candidate partition risks

There are **9,088 cross-partition image pairs**, projecting to **8,699 distinct
cross-partition ID edges**. Of those ID edges, **3,610 connect two different retained
partitions**; the other 5,089 connect retained data to existing quarantine.

| Candidate partition pair | Image-pair witnesses |
| --- | ---: |
| train/dev | 1,736 |
| train/final | 1,968 |
| dev/final | 106 |
| train/quarantine | 4,835 |
| dev/quarantine | 180 |
| final/quarantine | 263 |

The table counts image pairs; several image pairs can support one ID edge. Full
within-partition counts are recorded in `summary.json`.

| Graph | Components including singletons | Largest component | Cross-partition components | Members in crossing components |
| --- | ---: | ---: | ---: | ---: |
| Image-level dHash | 65,054 | 7,700 images | 544 | 10,137 images |
| ID-level dHash | 29,341 | 7,287 IDs | 313 | 8,280 IDs |
| Previous source/exact/identity components + dHash | 13,592 | 19,444 IDs | 95 | 19,766 IDs |

These are graph connectivity findings. Paths can accumulate weak visual-hash
similarities; members of a large component need not themselves be within radius 4
of one another and must not be relabeled as one person or one true duplicate.

## Isolation impact — proposal NOT adopted

For an auditable conservative impact estimate, the script proposes isolating the
entire closure of previous components plus dHash edges whenever that closure
crosses any candidate boundary, including existing quarantine. This avoids
splitting the prior source/exact/identity groups. It is deliberately not a minimal
cut and is not a verified data-cleaning decision.

That rule touches **19,766 IDs**, including all 3,034 already quarantined IDs,
and would additionally exclude **16,732 of 36,980 retained IDs (45.25%)**:

| Partition | Frozen candidate count, unchanged | Hypothetical added quarantine | Hypothetical remaining |
| --- | ---: | ---: | ---: |
| train | 33,302 | 15,927 | 17,375 |
| dev | 1,845 | 398 | 1,447 |
| final | 1,833 | 407 | 1,426 |

This giant closure and large exclusion burden make blanket adoption unsupported
without calibration and population analysis. **No exclusion was applied.** The
actual candidate counts remain **33,302 / 1,845 / 1,833**, with original quarantine
**3,034**. Frozen dev128 and old splits are unchanged. The suggestions remain
in an append-only evidence file for review; they are not replacement manifests.
Zero observed cross-partition edges after the hypothetical proposal is a graph
property, not a claim of formal readiness, semantic cleanliness or statistical
representativeness. Formal use remains blocked by calibration and other protocol
gaps; this audit does not attempt to resolve them by removing nearly half the data.

## Conditional top16 completeness logic

Read-only checks of the existing cache show:

- Retained nodes checked: **36,980**; retained nodes with kth cosine ≥0.9: **0**.
- Largest retained kth cosine: **0.8999828696250916**.
- Retained outgoing cached edges with cosine ≥0.9: **36,464**; cross-candidate-
  partition edges among these: **0**, including links to existing quarantine.
- Entire bank saturated rows: **356**, all outside retained data.
- Retained cached entries within `1e-5` of threshold 0.9: **31**.

**Conditional statement:** if the cached rows are the exact global top16 under
the same similarity computation, sorted in descending order, then kth cosine
strictly below 0.9 implies every neighbor scoring ≥0.9 is present. If an omitted
qualifying neighbor existed, the 16th-ranked score could not be below 0.9. This
argument applies to every retained node checked here. Saturated quarantine rows
do not by themselves refute this retained-row conditional statement.

The cache's rank completeness was **not recomputed** by an all-bank CPU search;
score precision and the same-score assumption matter near 0.9. This run checked
shape, ID coverage, sorted finite values, valid distinct non-self indices and
continuity with the previous audit's neighbor digest. The previous audit checked
cached cosine values against the normalized feature bank; it did not establish
global ranking completeness either. The result is a conditional combinatorial
coverage statement, not a biological identity guarantee or semantic clearance.

## Evidence files and independent verification

Remote paths below are relative to `R/protocol_near_duplicate_v1/results/`:

- `hash_members.jsonl`: complete hash-to-image membership, roles and candidate
  partitions; defines the image indices used in the compressed pair table.
- `hash_pairs.csv`: qualifying unique-hash pairs, same-hash buckets, exact
  Hamming distances and expected image-pair expansion counts.
- `image_pairs.csv.gz`: all 34,725 unordered image-pair witnesses with distance
  and exact SHA256 equality flags.
- `id_edges.csv`, `cross_partition_id_edges.csv`: all ID edges and crossing
  subset, with minimum distance and witness counts.
- `components.jsonl`: all three graph stages, memberships, partition counts,
  sizes and coverage flags. Image components use the documented image indices;
  ID components retain string IDs and leading zeroes.
- `risk_ids.csv`: complete crossing closure, including already quarantined IDs.
- `additional_quarantine_recommendations.csv`: 16,732 suggested additional
  exclusions, **not adopted**. No source data is moved or deleted.
- `summary.json`, `provenance.json`, `artifact_sha256.json`: counts, limits,
  assumptions, input/output hashes. `AUDIT_COMPLETE_NOT_FORMAL_READY` stores the
  summary digest and confirms enumeration completion only.

Total result size: **37,636,595 bytes**. SHA256:

```text
protocol_near_duplicate_audit.py
e8a4941f8d6110f9b99443ac423c2620aed09b41a8fbc6e418bbc12b83e7e4ee
summary.json
0ebe813e91d9dc95dd7a57142996b3a232e2dfd0f2de291835614ee21f01de67
image_pairs.csv.gz
8ec34ae44a1b8edbcdcb2fd61b84930427e3c1eadeb98af1ee37fb2a0436a5ee
additional_quarantine_recommendations.csv
3cb688c64103cfd98ad3f87fd5fc47344bdd45298d04ef59f313a65dd70b9664
```

Embedded tests passed locally and remotely, comparing indexed enumeration with
brute-force all-pairs on deterministic examples covering distances 0–5, block
boundaries and candidate deduplication. A separate read-only process returned
`independent_verification: PASS`, exit code 0, after verifying:

- Every output image pair's exact XOR distance and SHA256 equality flag,
  uniqueness, and complete Cartesian/combinatorial expansion of hash-pair rows.
- All **30,604** ID projections, minimum distances and witness counts against
  independently reconstructed image-pair projections.
- **128 deterministic sample hashes against all 78,822 hashes**: 10,089,216
  brute-force comparisons, with exactly matching emitted qualifying neighbors.
- Recommendation coverage of all crossing edge endpoints; recorded full-traversal
  counters; all input/output digests and completion-marker digest.

The candidate manifest and other inputs retained their hashes, checked again in
the independent process. No GPU or training process was queried or changed.

## Executed commands

Executed from the local workspace. Remote mkdir succeeded only because this
directory was new; do not redeploy over the completed evidence directory.

```sh
python3 -B .omx/experiments/m2h_ab_objective_v1/scripts/protocol_near_duplicate_audit.py --self-test
ssh -o BatchMode=yes -o ClearAllForwardings=yes 10.249.45.227 'mkdir /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_near_duplicate_v1'
scp -o BatchMode=yes -o ClearAllForwardings=yes .omx/experiments/m2h_ab_objective_v1/scripts/protocol_near_duplicate_audit.py 10.249.45.227:/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_near_duplicate_v1/protocol_near_duplicate_audit.py
ssh -o BatchMode=yes -o ClearAllForwardings=yes 10.249.45.227 "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -B /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_near_duplicate_v1/protocol_near_duplicate_audit.py --artifacts /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1 --out /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/protocol_near_duplicate_v1/results --max-seconds 210"
```

Limits remain explicit: audited hash coverage does not establish current image
immutability or coverage of every perceptual/semantic duplicate; dHash and raw
Glint are uncalibrated proxies; biological identity and source ontology are
unknown; historical checkpoint contamination, clean retraining and final protocol
approval remain outside this completed side task.
