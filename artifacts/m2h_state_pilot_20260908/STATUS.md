# M2H state pilot — 2026-09-08

User authorized first three short training arms and dev evaluation only.
State: HOLD_REBOOT — verified 2026-09-08 15:42 CST. Server rebooted during
health/preparation; all queue processes gone. DO NOT automatically relaunch.
queue.json still says running but is stale from the previous boot. Read this
STATUS and MONITOR_HOLD.json before any action. Previous ETA withdrawn.

Remote P=/data/muxiangyu/pythonPrograms/M2HImage
Remote batch=P/artifacts/m2h_state_pilot_20260908
Plan: .omx/plans/m2h-budget-first-plan-20260908.md

Frozen scope: C-H, C-perm, E-match; A2 step8400 common initialization,
A4 step8400 fixed teacher; rank16 original non-spatial architecture.
256 train-only source groups, two fixed train references each. Fresh M/I
conditions; H only as explicit training supervision or paired identity input,
never as mannequin conditions. Same teacher endpoint multiset for perm/match.
CF endpoint losses only; paired FM target epsilon-z_H only for paired branch.

Budget: preparation <=8 GPUh, first three arms <=60 GPUh.
Health20 updates (count toward arm total); fresh mean step<=80s =>300/arm,
80–120s =>200/arm, >120s =>stop/rebudget before efficacy results.
Atomic optimizer/model/per-rank RNG checkpoints every50, health20 saved too.
Reboot during health/preparation: stop expansion and diagnose, no relaunch loop.
No additional seed, 600-step extension, B training or full retraining authorized.

2026-09-08 14:12 CST: SSH working; boot13:28, GPUs idle, disk395G free.
last -x shows reboots13:02,13:10,13:28. Journal access restricted; cause unknown.
Historical experiment batches remain CLOSED. Do not rerun their evaluations.

## Launch evidence and operating instructions

- pool SHA256: 946260191282f1972ece34c5dd31517f2299403be8318339121716421c419b15
- Frozen256 source groups,32 train reference images,512 CF combinations.
  Excludes val/test source groups, dev128 M/I groups, audited exact duplicates.
- Cache READY:256 M +288 I (256paired +32CF) +256 H;285.664sec,0.07935GPUh.
 800 cache file hashes checked by training loader; Python undeclared-read bait denied.
- Queue PID13314; teacher PID13386 onGPU3; torchrun health PID13390 onGPU0/1/2.
 Boot ID c0d47840-12f5-4125-9126-4c79e98884c2. Do not duplicate launches.
- scripts/run_pilot.py owns bounded queue. queue.json is live machine state.
 health_C-H.log and teacher.log contain initialization/training/generation evidence.
- Health checkpoint step-008420 means20 NEW updates from A2step8400.
 C-H does not consume teacher files; it may run while teacher endpoints are built.
 At health20 choose300 if<=80sec/update,200 if80–120; >120 stops queue.
 budget_decision.json freezes before dev efficacy results. Shared LR5.625e-5,
 globalbatch18,micro1,accum6,rank16,seed0,uniformtau0..1,CFwindow.2–.8,
 identitydecodewindow.35–.7 every3 boundary,losscloth.5/invariance.2/id.1+.05,
 nohinge. Every CF branch recomputes ControlNet, teacher states detached.
- Queue then verifies512 endpoints/images and teacher overallimage/ID quality
 (never filters by scores), resumes C-H, runs C-perm/E-match same total,
 generatesdev128 each, three overall metric stages, paired report.
- Limits prepare8,train60,dev10 GPUh. Queue supervisor checks budget/disk/own
 child failures every5sec. It terminates only its own queued children on failure.
 Existing automation m2h updated in place, every2h; unchanged state quiet.
- Required launch-time verification pending: realmodel firstupdate, health20,
 active-weight audit, actualresume, finalthree-arms/dev report. CPU/mock tests
 passed9, plus real C-H cache800hash/shape preflight. No fullGPU test claimed yet.

## Latest verified progress — 2026-09-08 14:44 CST

- C-H has completed1 NEW optimizer update, globalstep8401. Firstupdate58.065sec,
 loss_total(rank0 average).2876689, finite loss/gradient checks passed.
 Identityattempt globally1, skips0; rank0-only identityloss can be0 while another
 rank triggers decode. Scalar loss logs are rank0, not all-rank loss means.
- Real checkpoint audit: active missing0, unexpected0, LoRAkeys/shapes/tensors
 matched. Four missing hair-only keys explicitly inactive and allowed.
- Teacher generated4/512, most recent22.92sec/image; detached float32 endpoints
 and PNGs written. Full512 hash verification still pending.
- Nine mock/CPUtests freshly rerun under actual refton torch2.7.1, all PASS;
 these do not substitute for actualhealth20/optimizerresume checks.
- Health20, checkpoint fsync+CPU audit, fresh-process actual trainingresume,
 C-perm/E-match,384dev outputs/metrics/report remain PENDING in queue.
- No further human confirmation required within authorizedthree-arms scope.
 If queue fails, inspect logs/state before recovery. Reboot is hard pause for
 expansion, not permission for automatic relaunch. Preserve partial results.

## Reboot incident — monitor 2026-09-08 15:39–15:42 CST

- last -x -F confirms new boots14:55:28 and15:08:23. Current boot ID
 42d48d90-5e25-4345-aef0-d00f7e585f41 differs from launch ID. No actual queue,
 teacher, torchrun, or training process remains; all GPUs idle. No relaunch made.
- C-H log has11 valid update rows endingglobal8411, mean54.981sec/update,
 observed identityattempt1/skip0. Tail contains1701 NUL bytes /1 invalid record.
 Actual later progress is unknown. No trainable.pt/READY checkpoint exists
 in new train outputs. Health20 and save were NOT reached/verified; no pilot
 weights or optimizer available for resume. These11 updates are not reusable.
- Teacher rows.jsonl contains29 complete JSON rows, not progress.json's stale28.
 All29 endpoint hashes, float32(3072,64), finiteness pass. PNG27/29 hashes and
 decoding pass; two PNGs are zero bytes:02574__id40970__seed0.png and
 42942__id32025__seed0.png. Preserve all originals; no deletion/regeneration.
 Partial teacher pool not READY and cannot start perm/match training.
- All800 M/I/H cache file hashes match sealed cache audit. Original frozen
 pool and cache retained; C-perm/E-match/dev evaluation have not started.
- No budget_decision.json:300 vs200 was NOT frozen. Do not promote the11-step
 throughput into a successful20-step health gate or declare training complete.
- Conservative budget charge until first new boot14:55:28 (launch14:39:25):
 teacher<=0.268GPUh, training<=0.803GPUh, cache0.07935GPUh; rounded total<=1.16.
 Includes initialization, not just optimizer time; precise interruption unknown.
 Disk394GiB available, above50GiB reserve. No ongoing GPU consumption.
- dmesg denied; journal limited to user-visible logs. Visible SSH port1995 bind
 conflicts do not establish reboot cause. No causal OOM/Xid/thermal diagnosis
 available. NUL/zero-byte files show persistence damage, not its hardware cause.
- Required next step: user/admin establish whether14:55/15:08 were deliberate
 reboots; if unexpected, inspect privileged kernel/OOM/NVIDIA Xid/thermal/power
 and hardware management logs. Existing permissions do not expose those logs.
 Only after stability issue is addressed and continuation directed should a
 bounded recovery be planned; no automatic GPU retry while this hold persists.
- Continue scheduled read-only monitoring; unchanged hold stays quiet. Preserve
 queue.json and interrupted logs as evidence; MONITOR_HOLD.json overrides their
 stale running labels. This is not a completed three-arm experiment.
