# M2H A/B execution ledger

User authorized execution and monitoring every two hours on 2026-09-07.
Plan: `.omx/plans/m2h-objective-experiments-20260907.md`.

## Monitoring

Codex thread heartbeat id `m2h`, ACTIVE, every 2 hours. Check the calling
thread's automation before creating another; repeated user request is one
monitor, not two. Quiet on unchanged progress; report completion/failure,
material metric regression, budget breach, or required user action.

## Current phase

P0 protocol audit and B-D0 input-side diagnostic implementation.
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
