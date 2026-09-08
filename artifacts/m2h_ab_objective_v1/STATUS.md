# M2H A/B execution ledger

User authorized execution and monitoring every two hours on 2026-09-07.
Plan: `.omx/plans/m2h-objective-experiments-20260907.md`.

## Monitoring

Codex thread heartbeat id `m2h`, ACTIVE, every 2 hours. Check the calling
thread's automation before creating another; repeated user request is one
monitor, not two. Quiet on unchanged progress; report completion/failure,
material metric regression, budget breach, or required user action.

## Current phase

FINAL UPDATE 2026-09-07 17:30 — supersedes ALL running snapshots below.

- A-D0 COMPLETE:1296 predictions, all image/ID/pose passes complete;3 missing
  output faces penalized -1, no metric infrastructure failures.
- B128 COMPLETE:512 outputs and all3 metrics, all512 faces detected;
  ECC80/128 accepted,48 fallback,0 method failures;64M x11 calibration cases.
- Full `a_d0_full_v2/READY` and `a_d0_b_report_v2/READY` verified. Supervisor
  PID175467 is no longer running. No matching GPU/experiment processes remain.
- Full pipeline3022.343s; GPU child-process wall time including startup2.872822h,
  excluding earlier smoke/cache/B32.61 targeted regression tests passed.
- Objective findings: A matched raw ID advantage has large carry-in confound,
  not a trained-method gain. B HF-LPIPS base .142070, inplace .068963,
  warp-feather .078730, warp-multiband .077399; warp controls did not exceed
  strong inplace copying. Boundary proxies failed reliable perturbation ordering.
- DATA AUDIT CLOSEOUT COMPLETE, but `formal_split_ready=false` persists:
  candidate source/identity/hash split is not a certified clean final protocol.
  Further near-duplicate risk adjudication would be a NEW work batch; the
  16,732 additional exclusions were NOT adopted. Do not call P0 fully passed.
- Final Chinese report: `EXPERIMENT_REPORT_20260907.md`; machine-readable
  report remote `a_d0_b_report_v2/summary.json`, local mirror
  `a_d0_b_report_v2_summary.json`. All original code/data/splits preserved.
- CURRENT REQUESTED COMPUTATIONAL BATCH CLOSED. No A/B parameter training,
  additional audits, reruns, or threshold changes should be launched by the
  heartbeat for this completed batch. Keep quiet on unchanged finished state;
  wait for the user's next experiment request before expanding scope. Preserve
  existing two-hour monitoring configuration; no duplicate automation needed.

Heartbeat verification 2026-09-07 17:33:05 +08:00: both pipeline/report READY
markers remain present; pipeline summary unchanged (1296 A predictions,128 B
pairs,training_steps0,formal_split_ready=false). No GPU compute processes or
matching experiment processes. No jobs launched, no scope expansion, no repeat
completion notification; previous final report remains current.

Historical execution snapshots (not current job state):

Execution update 2026-09-07 16:34 follows for reproducibility only.
Full duplicate audit is COMPLETE (80,028/80,028; 111 exact groups, 20 cross-split).
New requested scope: A-D0 + protocol closeout + B alignment/boundary controls.
No learned A/B training is authorized by this diagnostic phase.

- Data `data_protocol_v2` frozen: 64 M, 32 reference files, 128 pairs, max raw
  Glint M/ref cosine .280375 (<.3), seed0; M/I file pools disjoint. SHA256
  `9cc9bc517e65486c2794cde0ef1d002d8db0f3f186bc2adc0c8fcdc5843929be`.
  These are reference FILES, not 32 verified biological identity clusters.
  v1 retained but superseded: it over-reused only3 reference files and was
  rejected before A generation. Formal clean split is still NOT ready.
- `diagnostic_cache128_v2`: READY; 64 M +64 H_supervision +32 I caches.
  Cache generation29.776s GPU2; peak2.106GiB. H only initializes original-state
  diagnostic, never an M/I condition. I cache rebuilding remains formal gap.
- `a_d0_smoke1_v2`: small run ACTIVE at this snapshot GPU0, 1M x2refs,
  expected36 prediction rows. No full A run launched before smoke validation.
- `b_alignment32_v1`: COMPLETE128 outputs, 24/32 accepted ECC,8 identity
  fallback,0 method errors. CPU80.365s. Independent dev ID, image, pose passes
  completed. Raw mask-crossing boundary score is invariant for eroded edits.
- `b_alignment32_v2`: COMPLETE, same4 rendering methods plus FIXED input-M
  distance<=16 transition band and feathered synthetic perturbation calibration.
  CPU118.864s,24 accepted/8fallback/0 failures. v1 retained for provenance.
- Tests:26 protocol/B tests passed remote;19 B tests passed after transition
  scoring revision. No package installs/downloads, no AdaFace, no G2/human labels.
- Protocol side audit COMPLETE `protocol_group_audit_v1/results`:
  corrected source/exact/top16>=.9 graph19,303 components,max282 (not legacy17036).
  Candidate train33302/dev1845/final1833/quarantine3034; all40014 IDs exactly once,
  zero serialized evidence edges crossing candidate partitions. Leader independently
  reloaded all CSVs and verified output digests. No original files changed.
  Formal readiness remains false; see DATA_PROTOCOL_CLOSEOUT.md for limitations.
- Near-duplicate follow-up COMPLETE `protocol_near_duplicate_v1/results`:
  all80,028 audited image hashes, exact enumeration of all Hamming<=4 pairs,
  34,725 image-pair candidates,3,610 ID risk edges crossing retained partitions.
  These are dHash risk proxies, NOT verified duplicates/identities. Wholesale
  union would produce a19,444-record giant and suggest16,732 additional
  quarantines; do NOT silently adopt that major exclusion. Risk manifests are
  supplementary only. Candidate split and dev128 were not changed.
  Runtime27.854s CPU. No further audit expansion needed in this A-D0/B scope;
  formal protocol gate remains withheld, not failed computational execution.
- Full supervisor PID175467, physical GPU0/1/2/3. Check command, not PID alone.
- 17:21 A generation and all3 objective passes COMPLETE:1296 predictions,
  3 output_face_failed retained with -1 penalty; all image/pose computations OK.
  B128 ACTIVE on shared teachers. Do not rerun A or start training.
- Final report command AFTER full pipeline READY: `OPENBLAS_NUM_THREADS=4
  /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
  /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/scripts/summarize_a_b_v2.py
  --run /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/a_d0_full_v2
  --out /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/a_d0_b_report_v2`.
  Fresh directory required;4 summary regression tests passed. Report raw output
  identity AND carry-in adjusted change; bootstrap clusters source/ref-file proxies,
  not biological truth. No A/B training automatically follows this diagnostic.

16:37 update: smoke COMPLETE36/36 predictions, all36 image/ID/pose rows OK.
Generation298.287s,peak31.168GiB. `a_d0_full_v2` supervisor now LAUNCHED.
Command from remote R: refton_m2h Python `scripts/run_a_d0_and_b.py --repo
/data/muxiangyu/pythonPrograms/M2HImage --artifacts
/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1 --cache
/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/diagnostic_cache128_v2
--out /data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1/a_d0_full_v2`.
Log `a_d0_full_v2_supervisor.log`; exact child commands/PID are recorded under
`a_d0_full_v2/manifest.json` and `logs/*.command.json`. Check live process,
per-shard progress, logs and FAILED.json before doing anything; do NOT relaunch.
Supervisor handles4 GPU shards then all metrics and B128
on exact shared teachers. It creates READY only after all stages; failures
remain explicit. Do not infer completion from scripts alone.

No new long training has been started. No G2/manual annotation gates.
Do not claim the old 400-image test is a new final test.

Remote repo: `/data/muxiangyu/pythonPrograms/M2HImage` on `10.249.45.227`.
Remote dataset: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2`.
New remote execution root: repo `artifacts/m2h_ab_objective_v1`.
Python: `/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python`.

Existing repo is dirty; preserve all existing code/results. Initial HEAD
`c91d2486bdde29df35d40832f326658d8261e513`; exact diff snapshot is in P0 audit.

## Work and gates

- P0: source metadata/split membership audit, exact-hash candidate check,
  identity-group coverage audit, input-only condition audit; group split
  remains unready until all grouping and provenance checks pass.
- B-D0: deterministic source-only VAE roundtrip and automatic M-side parsing
  on fixed old-val candidates, not a learned method experiment.
- Next: implement input-only condition namespace and 32-example inference
  access test; calibrate objective metrics and freeze clean protocol.
- Only after P0 gates: A-C0/C1/C2/E pilot and B copy/warp controls.

First-round proposed budget cap is 640 GPUh (P0 80, A 400, B 160).
Long training advancement requires plan gates; monitor must not start jobs
blindly, change thresholds after seeing candidate results, or stop unrelated
processes. Record each exact launch command, PID/session, output directory,
elapsed time and actual GPU count below as it becomes available.

## Completed evidence (2026-09-07, Asia/Shanghai)

- P0 metadata/hash audit: 40,014 unique IDs, split lists match manifest;
  22,777 source-parent candidates, 1,571 crossing original splits, involving
  11,551 rows. SHA256 checked both human and mannequin images for these
  candidates: 0 cross-split exact duplicate groups, 0 read errors. This is
  NOT a full-corpus near-duplicate/biological-identity clearance. Runtime
  232.605s, CPU-only. Remote `p0_metadata_20260907/summary.json`;
  local mirror `p0_summary.json`.
- B-D0 smoke4: 4/4 successful, 12.163s on GPU3.
- B-D0 oldval64: 64/64 parsing/reconstruction success, 60.195s on GPU3;
  garment PSNR mean 36.2943dB, garment MSE 0.0003541254, HF-MSE
  0.0002746311 (sigma=1.5, NOT historical HF-LPIPS).
- Same fixed64 with locally cached AlexNet LPIPS 0.1.4: 64/64 success,
  47.507s on GPU3; garment bounding-box LPIPS mean 0.00744828. This is
  unmasked bounding-box LPIPS, not a garment-interior-only/perceptual
  truth metric. Results do not establish VAE as the dominant bottleneck.
  Remote `b_d0/oldval64_lpips/summary.json`, local mirror
  `b_d0_oldval64_lpips_summary.json`.
- input-only v1 smoke2: both outputs generated, 301.152s on GPU0,
  peak allocated 31.149GiB. Original script retained remotely as
  `scripts/input_only_probe_v1.py`.
- guarded smoke1: output generated, 75.768s on GPU0, peak 31.149GiB.
  Target bait read denied exactly once; no additional denied reads and no
  subprocess attempts; 9 allowed dataset read events. This proves tested
  Python path/subprocess contract only, not arbitrary native syscall
  isolation. Remote `input_only_guard_smoke1/audit.json`; local mirror
  `input_only_guard_smoke1_audit.json`.
- Identity feature smoke16: 16/16 successful, StableAnimator Python,
  ONNX CUDA providers explicitly confirmed. 1.165s extraction time; model
  initialization is additional. Earlier imagdressing smoke failed before
  extraction because its ONNX runtime is CPU-only; failure artifact kept
  in `identity_features_smoke16`, recovered using existing GPU environment
  without installing/changing dependencies.
- Tests: p0 5 unittest tests, input-only 5 unittest tests, B-D0 14 pytest
  tests. Local machine lacks NumPy; NumPy-dependent tests use remote conda.

## Background job registry

All relative remote paths below are under
`/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_ab_objective_v1`.
Do not relaunch while matching processes are alive. PIDs may be reused:
always verify command line and output path as well as PID.

### Identity audit features — COMPLETED

- PID at launch: `164413` (detached nohup Python).
- GPU: 1, extraction timeout guard 8 hours; no model training.
- Log: `identity_features_full.log`.
- Artifacts: `identity_features_full/manifest.json`, `progress.json`,
  `per_image.jsonl`, `embeddings_*.npz`, final `summary.json`/`FEATURES_READY`.
- Last checked: 11,200/40,014, zero failures at that point; not a final count.
- Command (run from project root):

  `/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python -u artifacts/m2h_ab_objective_v1/scripts/identity_group_features.py --root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2 --out artifacts/m2h_ab_objective_v1/identity_features_full --device 1 --max-hours 8`

- Purpose: same all-split ArcFace preprocessing for grouping; not final
  identity evaluation. Failures remain explicit and require quarantine;
  completed features do not automatically constitute a clean split.

### Guarded input-only inference32 — COMPLETED

- Launch supervisor PID: `165530` (`timeout 7200s`, detached nohup).
- GPU: 0; hard wall-time limit 2 hours. Child Python PID is discovered from
  process command line, not guessed.
- Log: `input_only_guard32.log`.
- Artifacts: `input_only_guard32/audit.json`, `M/`, `I/`, `masks/`, `outputs/`.
- Command (run from project root):

  `/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -u artifacts/m2h_ab_objective_v1/scripts/input_only_probe.py --repo /data/muxiangyu/pythonPrograms/M2HImage --root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2 --ids-json artifacts/m2h_ab_objective_v1/p0_metadata_20260907/old_val_probe_ids.json --out artifacts/m2h_ab_objective_v1/input_only_guard32 --limit 32 --device cuda:0 --generate`

- Run environment: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- New M mask/garment features, M-side pose, null head token; disjoint 32 M
  and32 I file pools from old-val. I cache only selects identity-side keys.
  Dataset read allowlist rejects target/old M caches, subprocesses denied.
  Native library syscall isolation and fully regenerated I provenance are
  still gaps; do not mark the formal input-only gate passed solely from
  these generated outputs.

## Next monitor actions / continuation

1. Inspect the above jobs and logs before creating any new task; copy final
   summaries locally and account GPU wall-time including startup/failed runs.
2. Check inference32 output count=32, finite arrays, guard denied_reads only
   the one deliberate bait, no denied_subprocesses; failures are not skipped.
   Add native file-access tracing if an existing tracer is available; do not
   claim stronger isolation than tested. Original target-derived M cache
   fields must never be used as a silent fallback.
3. Finish all-role identity similarity/group audits and near-duplicate/source
   union, quarantine unresolved IDs; thresholds/dev/final still not frozen.
   Source-parent overlap alone is not biological-identity ground truth.
4. Run copy+blend/warp controls on new input-only base outputs and input-fixed
   masks, before training B. Compare detail plus boundary/identity/pose;
   low VAE-only LPIPS makes the VAE-bottleneck hypothesis tentative.
5. Resolve independent dev recognizer availability without using final
   AdaFace for selection. Scoped searches were incomplete, not a proof of
   global absence. No dependency installs were done.
6. Only when P0 gates pass, implement/launch A matched/mismatched-state
   pilots and B learnable controls. Do not start old G2/human annotation
   work, old watcher manual review gates, or train from an unclean split
   while labeling results formal. Keep the 640 GPUh exploratory cap.

Server `/data` had about399GB free at initial check (95% utilized); verify
free space before large teacher/checkpoint caches, preserve existing files,
and reserve at least50GB rather than deleting historical experiments.

## Heartbeat execution update — 2026-09-07 14:54 trigger

This section supersedes earlier running snapshots.

### Completed and verified

- `identity_features_full`: 40,014/40,014, zero failed, 864.097s extraction,
  GPU1. Features complete, not final group split. Local summary mirrored.
- `input_only_guard32`: 32 outputs verified readable 768×1024; all64 M/I
  NPZ caches finite; only1 deliberately denied target bait read; no denied
  subprocess attempts. 1453.381s on GPU0, allocated peak31.149GiB. This
  run finished normally, not timed out. Local audit mirrored.
- `input_only_native_trace1`: another1-example run with native strace
  open/openat tracing, 74.854s GPU0. `native_trace1_check_v2.json` reports
  12 successful dataset opens, no unexpected paths or unresolved trace
  entries. The first checker flagged checkpoint-directory enumeration;
  v2 explicitly permits only that declared directory. Both reports retained.
  This validates the observed one-example native file path, not all future
  execution or newly regenerated identity-side cache provenance.
- `copy_controls32`: 32 pairs×4 methods, zero failures, CPU42.438s. Fixed
  M-derived masks, same ROI for all methods. Mean interior HF-MSE:
  base .00378815; feather(e4,f8) .00000916562; feather(e8,f16) .000202750;
  identity-aligned Poisson .000595876. Feather changes zero pixels outside
  allowed region; Poisson outside-change fraction .0349102. Source-boundary
  gradient difference is only a diagnostic proxy, NOT a defect-rate gate.
  This strong copy baseline prevents claiming ordinary detail preservation
  itself is novel. Warp/multiband and calibrated boundary/pose tests remain.
- Independent dev recognizer FOUND and exercised without downloading or
  installing: `/home/muxiangyu/.insightface/models/buffalo_l/w600k_r50.onnx`,
  SHA256 `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
  This replaces the planned but unavailable MS1MV3-R100 development weight
  before any new A/B learned pilot; it is distinct from training Glint-R100
  and final AdaFace. Exact upstream training data/license provenance must
  be verified before publication; filename alone is not that verification.
- `copy_controls32_dev_identity`: all32 input refs eligible and all128
  outputs detected. Dev cosine means: base .481294; feather4/8 .478084;
  feather8/16 .482025; Poisson .461898. These are exploratory point estimates,
  not significance/noninferiority verdicts and NOT comparable to historical
  AdaFace means. Runtime4.435s GPU2 excluding init; weights/entrypoint frozen
  in its manifest. AdaFace was NOT called.
- `identity_neighbors_full`: full40,014×40,014 cosine matrix evaluated in
  blocks, top16 retained per row. Median nearest cosine .889134. At exploratory
  threshold.9, 5,431 rows have a retained neighbor in another original split.
  This is high-similarity evidence, not verified same biological identity.
- `group_sensitivity`: combining top16 identity candidates and source-parent
  candidates produces giant connected components: at .8 largest34,638;
  at .9 largest17,036. DO NOT naively freeze a split from this heuristic or
  claim these are real identity-group counts. Top-k truncation, parent-key
  semantics, threshold calibration and transitive bridges need examination.
  If genuine identity–garment graph connectivity prevents useful joint
  holdouts, propose explicit bridge quarantine and separate identity/garment
  generalization tracks; do not silently waive disjointness or discard most
  samples. No new split/data deletion has been made.

### Current live job

- Full dataset exact-SHA/perceptual-hash candidate audit, CPU4 threads.
- Supervisor PID `167412`, `timeout 7200s`, nohup, no GPU reservation.
- Remote log `full_duplicates.log`; outputs `full_duplicates/`.
- Last check47,500/80,028 image files (both human and mannequin), zero read
  failures at that point. This extends earlier parent-candidate-only audit.
- Exact launch from P:

  `/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -u artifacts/m2h_ab_objective_v1/scripts/full_duplicate_audit.py --root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2 --out artifacts/m2h_ab_objective_v1/full_duplicates --workers 4`

- `summary.json`, `exact_groups.json`, `dhash_equal_candidates.json` and
  `failures.json` on completion. Equal64-bit dHash is only a candidate;
  nonzero-Hamming near-duplicates are not covered by exact hash equality.
- Next wake: check this process/output before rerunning; verify output
  counts, examine cross-split exact matches and quarantine/group proposal.

### Accounting / next work

Recorded successful GPU execution durations sum to roughly0.80 GPUh;
model startup not included in some feature/evaluator scripts, so treat this
as a lower bound, not exact billing. No learned A/B training has started.
First-round640 GPUh cap is unchanged. Remaining pretraining gates: calibrated
clean split with bridge handling, independent-dev protocol freeze, input-only
identity-cache rebuilding/provenance, calibrated multi-metric baseline.

Do not continue searching for a dev recognizer as though none were found;
the exact frozen candidate above is now available. Next implement warp and
boundary/pose controls as needed, and examine grouping evidence before
launching long training. Existing historical code/results remain untouched;
all new scripts/results are in the separate artifact namespace.
