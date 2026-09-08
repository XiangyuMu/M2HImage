# A2 pure-input supplementation and clothing localization

## FINAL — verified complete 2026-09-08 13:34 +08

This supersedes ALL running/outage/recovery snapshots below. Authorized first
priority batch CLOSED: no remaining model generation/region evaluation/report
tasks, no automatic training or reruns. Existing heartbeat should stay quiet on
unchanged completion and must not restart earlier recovery scripts.

Authoritative remote final: N/report_cpu_v5, N/assembly_cpu_v5/READY. Local mirror
report_cpu_v5/ beside this STATUS. N denotes P/artifacts/m2h_a2_localization_20260908.
All8 CPU shards completed before another server reboot at13:28:19. Durable
outputs survived: each240 region rows+120cross-ref rows, all OK; assembly_v5
only verified/merged/reported these outputs, NO additional model calls.
Main A2 generated128, reused B2/A4 each128. A2 image/ID/pose128 each verified,
384PNG hashes verified. Final1920 regional rows+960cross-reference rows,
384regional images and192cross-reference pairs. All source regions eligible
in128 main pairs. Sensitivity101 preserved; cross-ref sensitivity requires both
references retained and thus43M. Smoke/full CPU metric differences<=1e-6.
Final whole-garment CPU/GPU max abs differences: DINO1.370907e-6,
HF-LPIPS5.394220e-5. Same CPU backend used for all three regional arms;
ordinary table retains original GPU scores, not bitwise-equivalence claim.

Report/summary/budget local SHA verified against COMPLETE:
- REPORT.md: 5bacf3325a5d32aee68ccdd8734e811c4a198725cf2adb2fbaf07a5911357706
- summary.json: fe832e07cdcec38c2133d914e9b7795b10829efbf700a181c8af5c5a5a2157e3
- BUDGET.json: cbf1e51787d5c4e0b995c377bf93edbf6c1232404d0dbf9c62e1033efad07b47
Independent bounded code review found no must-fix CPU-path issue.23 prior
targeted tests passed after reboot;9summary regression tests explicitly rerun
against summarize_cpu module passed. Final process inspection found no region/
assembly/recovery jobs running. No parameter training, no manual labels/G2.

Key results, exploratory paired10000-draw source/ref bootstrap:
- A2-B2 ID+.019821 CI[.007734,.032554], DINO+.014140 CI[.003117,.026273],
  HF-LPIPS-.017541 CI[-.029450,-.006673]. Body/head intervals cross0.
- A4-A2 ID+.078110, but full garment DINO-.055959 and HF+.077470.
- A4-A2 regional HF degradation: interior+.103460 CI[.081956,.125863],
  boundary+.038810 CI[.028072,.051048], hightexture+.077907 CI[.053032,.102460].
  Interior native-gradient MAE+.007639 CI[.004520,.011608], corroborating
  that the deficit is not confined to the boundary/masked resize artifact.
  Do NOT compare absolute magnitudes between masks to rank damage severity.
- These source-fidelity directions remain in101 sensitivity pairs. Cross-ref
  hightexture drift A4-A2+.038021 CI[.010500,.064551]; whole-garment drift
  +.021524 CI[-.013891,.063588] is inconclusive, not a global leakage claim.
- A2/A4 differ in multiple training factors; no isolated causal attribution
  to identity loss, sampling, or state conflict. Hightexture is NOT OCR truth;
  boundary metrics do not establish naturalness or anatomical correctness.

Budget: CPU recovery added0GPUh, longest shard503.4s,8shards×4CPUthreads.
Recorded A2/old-smoke GPU wall plus conservative bounds for two lost GPU tasks
total1.2338GPUh (<12cap), NOT exact all-inclusive timing. All failed/empty
historical folders retained. Server reboot cause remains unknown; CPU run also
followed by reboot, so do not claim GPU evaluation caused previous failures.

## Latest — 2026-09-08 13:20 +08 CPU recovery v4 running

User explicitly requested redo regional evaluation/report. At13:11 connection
restored, boot13:10:02 confirmed, no experiment/GPU process survived. CPU-only
fallback chosen (no GPU/root-cause claim). First2M CPU smoke completed60 image
region rows and30 cross-ref rows,0 failures; no images regenerated.
DINO vs original GPU max discrepancy1.371e-6; HF-LPIPS5.394e-5. Thus strict
old1e-5 all-metric equality failed; cpu_recovery_v3 stopped BEFORE full launch,
preserved. For CPU-only regional analysis, before full results, disclosed bounds
frozen DINO1e-5/HF-LPIPS1e-3; this is a backend numeric check, NOT efficacy gate.
All three arms use same CPU metrics; original overall GPU table remains separate.

New supervisor scripts/run_cpu_recovery_v4.py, cpu_recovery_v4 and target
report_cpu_v4. Eight disjoint8M shards, each4 CPU threads, 0 GPU usage. Reuses
cpu_smoke_v3 and384 valid predictions; full output1920 region/960 cross-ref rows.
PID and commands in cpu_recovery_v4/manifest.json and logs/. All old failed/empty
artifacts retained. Check cpu_recovery_v4.log/FAILED before action; no duplicate
launches or further GPU attempts. Original scripts unchanged; new summary is
scripts/summarize_cpu.py. Reports fsynced before final completion.
Earlier snapshots below superseded. No parameter training authorized.

## Latest — 2026-09-08 about13:10 +08, connection lost again

After restoration and successful384-image preflight/23tests, recovery PID7165
was last confirmed in smoke stage. SSH then timed out again on multiple calls,
launcher disconnected and ping received no response. Completion unknown.
Do NOT launch another recovery or infer prior process died from connection loss.
Repeated disconnect occurred near regional metric startup, but cause is UNKNOWN:
no kernel/OOM/GPU/network logs retrieved, no causal GPU-failure claim warranted.
On connectivity restoration first inspect recovery_v2.log and live PID/command;
if idle or failed, retrieve system journal/kernel prior-boot errors and resource
evidence before any additional GPU launches. No system/network settings changes,
no new dependencies and no other users' process termination authorized.
Existing two-hour heartbeat follows this STATUS. Notify recovery/new evidence,
stay quiet on unchanged already-reported outage. This batch is NOT complete.

## Recovery attempt — 2026-09-08 13:09 +08

Server restarted13:02:38+08; connection restored. Fresh preflight rehashed384
PNGs and verified all A2 metric keys/paths/finite fields successfully. No image
regeneration needed. Both old region smoke and full per-image files are0bytes,
despite old smoke READY: old region outputs are unusable, retained unchanged.
No prior experiment/GPU process survived. New recovery PID7165 uses
`scripts/recover_localization.py`, `localization_recovery_v2`, and target
`report_recovered_v2`. It reruns first2M region smoke and full64M only,
then paired statistics and budget. 23 targeted tests passed fresh after reboot.
Lost full task time conservatively bounded by preserved old smoke launch to
reboot; recorded original A2/smoke costs retained. Cap12GPUh unchanged.
Check `recovery_v2.log`, `localization_recovery_v2/manifest.json`, FAILED and
final report COMPLETE plus row counts/hashes. Do not relaunch running recovery.

## Historical — 2026-09-08 12:52 +08, connection lost; NOT batch complete

A2 completed128 PNGs and image/ID/pose metrics128 each. Supervisor10420 exited.
Observed ordinary A2 means: ID0.5278678767, DINO0.8544213586,
HF-LPIPS0.3223486845, body0.0390704829, head0.0300556383.
These are point estimates read from remote metric summaries; paired report
not retrieved yet. No training.

Localization supervisor PID12478 was confirmed. At12:48 first2M smoke passed,
full64M directory existed and full regional process was launched. Since about
12:49 SSH repeatedly timed out and ping returned no packets; launcher session
ended with broken pipe. This is NOT evidence the remote job stopped. Do not
relaunch on reconnection: inspect PID/command, localization_v1/manifest.json,
logs/full64M.log, FAILED.json, report_v1/COMPLETE.json and full counts/hashes.
Full process may finish independently under nohup. Region/report script hashes
are pinned by the live supervisor; do not edit them while it runs.

Independent verifier PASS for preflight/converter/source-only code contracts;
23 remote tests passed. No live-run final verdict due connectivity gap.
Existing heartbeat m2h updated in-place to this new batch, every2hours, no
duplicate automation. Quiet on unchanged outage; notify meaningful recovery,
completion or new failure. Task remains pending retrieval/verification.

On reconnection: finish final verification and mirror report_v1 locally. If
report failed, preserve artifacts and correct in fresh versioned report/run;
never count failed or partial metrics as successful. The authorized scope is
only this first-priority batch, not subsequent A/B training.

Authorized 2026-09-08: A2 same128 pairs plus reuse existing B2/A4 for objective
localization. No parameter training, no G2/manual labels, no historical overwrites.

Remote P=/data/muxiangyu/pythonPrograms/M2HImage.
New batch P/artifacts/m2h_a2_localization_20260908.
Previous R=P/artifacts/m2h_minimal_revalidation_20260907 remains CLOSED.

## Frozen scope before new results

- A2 checkpoint phase2_a2_diff_r16_4000_768x1024/checkpoints/final, step8400.
- Shared R/fresh_cache_v2; original dev128 pairs SHA
  9cc9bc517e65486c2794cde0ef1d002d8db0f3f186bc2adc0c8fcdc5843929be.
- Same seed0, Euler20, pure M/I conditions. Four disjoint32-pair shards,
  all using unchanged hash-pinned prior generator through strict A2 wrapper.
- Reuse128 B2 and128 A4 pure-input predictions and ordinary objective metrics;
  record image/manifest hashes. No historical legacy-M generation.
- New region diagnostics fixed by M only: interior eroded8px at768 width,
  inner boundary complement, top25% source-gradient hightexture and complement.
  Report support/empty masks and all output failures. No output-based eligibility.
- Regional DINO/HF-LPIPS and same-M cross-reference clothing drift, diagnostic
  only. Position changes may affect scores; no boundary-naturalness claim.
- No OCR/text correctness claim without a separately validated source-only OCR
  eligibility protocol. Hightexture is not a semantic text/print ground truth.
- Pairwise comparisons B2/A2/A4, full128 plus prior101 sensitivity subset;
  exploratory M-source/ref-file clustered intervals, not training-seed evidence.
- Generation/process cap2h each, new batch cap12 GPUh, reserve50GiB storage.
  Region runtime measured and bounded separately before expanded passes.

## State

Generation supervisor PID10420, run_v1, launched2026-09-08. Four32-pair
shards on GPUs0-3; actual child commands/PIDs in run_v1/logs/*.command.json.
Wrapper rejects active adapter/LoRA key or shape mismatch before sampling;
only four historical inactive hair keys may be absent. No current batch success yet.
Strict compatibility succeeded at step8400 with no active missing keys. Tests:
5 checkpoint +9 regional partition/contracts +9 paired statistics/conversion
passed in remote refton_m2h environment (23 total; no installation).
Localization finisher launched in localization_v1, waiting A2_READY then does
independent strict preflight of generated keys/hashes and all3 metric stage
keys/paths/status/finite values. Inspect localization_v1/manifest.json for PID.
It runs first2M region smoke then full64M, verifies same-subset reproducibility,
and writes report_v1 plus budget and final COMPLETE. It never trains/regenerates.
Region records are long-form:1920 per-image rows (384images×5regions),960
cross-reference rows (64M×3arms×5regions). Same-M drift requires both reference
pairs in sensitivity subset, so its subset count may be smaller than58M.
One agent copied intermediate region scripts under remote P/.omx/experiments;
these are unused code copies, not additional experiment runs. Authoritative
scripts/run directories are under P/artifacts/m2h_a2_localization_20260908.
Inspect fresh process command/logs before any launch; no duplicate launches.
Completion requires A2 image/ID/pose128 rows each, all384 regional inputs checked,
paired statistics and final hash-verified report. READY from generation alone
does not mean this batch is complete.
