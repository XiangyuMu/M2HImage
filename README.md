# M2HImage FLUX Phase 1 Warmup / B2' / A2 / A4 / Inference Diagnostics

This repository contains the FLUX.1-dev MA-RA-CDT paired warmup, B2' adapter-only baseline, A2 differential counterfactual experiment, the preregistered one-shot A4 identity-directed gate, and the post-hoc inference-only velocity, garment-protection, trainable-weight interpolation, and identity-condition interpolation diagnostics.

## Current Project Snapshot (2026-08-17)

The original A2/A4 mechanism sequence and its preregistered verdicts remain frozen. A new, separately registered spatial-conditioning study is active to repair the weak garment, hair, and head-direction input routes without rewriting those results. The tile zero-shot probe and the 6144-token rank-16 A6000 VRAM gate have passed; the spatial cache and repaired warmup are the current execution path. Code and compact reports are tracked in Git; datasets, caches, generated images, checkpoints, and large evaluation artifacts remain on the local data volume and are not tracked by Git.

### Final Experiment Results

All decision comparisons passed the fairness-field checks for start checkpoint, sampler state, train IDs, seed, global batch, learning rate, LoRA rank, schedule, and continuation length. B2' is a reference baseline; A2 and A4 are judged against the same equal-step B2-cont control.

| run | held-out sim_target | DeltaID | GarmentSim | pose cross-ID variance | verdict |
|---|---:|---:|---:|---:|---|
| B2' adapter-only reference | 0.4196 | 0.4063 | 0.8622 | 89.3842 | identity baseline PASS |
| B2-cont, paired-only +4000 steps | 0.4223 | 0.4088 | 0.8951 | 90.5637 | equal-step control |
| A2 differential +4000 steps | 0.4388 | 0.4201 | 0.9016 | 89.0563 | FAIL on preregistered garment gate |
| A4 directed identity +4000 steps | 0.4944 | 0.4794 | 0.8778 | 88.7023 | MIXED identity-garment trade-off |
| A4 protected all-step, pure inference | 0.4942 | 0.4788 | 0.8663 | 88.0894 | NULL: gap recovery -66.49% |
| A4 protected tau=[0,0.2], pure inference | 0.4944 | 0.4794 | 0.8777 | 88.7433 | NULL: gap recovery -0.68% |

Frozen decisions:

- B2' established that pretrained PuLID identity injection works: `sim_target=0.4196`, `DeltaID=0.4063`, 400/400 valid faces.
- A2 improved GarmentSim over B2-cont by only `+0.0065` with one-sided Wilcoxon `p=0.066480`; bottom-quartile gain was `+0.0063`, below the preregistered `+0.02` threshold. The A2 garment-axis verdict is `FAIL`.
- A2 diagnosis found the differential losses `BOUND`, resolved `hinge_g=0.0291450452`, and a significant held-out DeltaID gain of `+0.011327` (`p=2.6466e-6`). This justified the single A4 identity-axis run, but does not reverse the A2 garment verdict.
- A4 increased held-out `sim_target` over B2-cont by `+0.072096` and DeltaID by `+0.070592`, both with greater-side Wilcoxon `p<1e-8`. Identity treatment passed strongly.
- A4 GarmentSim regressed from `0.8951` to `0.8778`; the deterioration test gave `p=0.00102455`. Pose variance, face detection, and detector-confidence realism did not regress. The final preregistered verdict is `MIXED`.
- Semi-hard sampling was measurably stronger: mean training-recognizer distance increased from `0.9420` for the replayed A2 random policy to `1.1000` for A4. A4 identity-loss face-detection skip rate was `1.95%`; training `sim_gap` rose from `0.0787` in the first quartile to `0.1599` in the last quartile.
- Qualitatively, identity response and image clarity are healthy, but garment conditioning often remains generic or mismatched. The metric regression confirms this is a real trade-off, not only a visualization artifact.
- The post-hoc velocity probe completed 20/20 stratified trajectories. For `Delta v_io`, the normalized energy split was face `15.19%`, cloth-safe `14.00%`, body/background `58.67%`, and other `12.14%`. Cloth energy exceeded the fixed 10% feasibility threshold, so the protected sampler was evaluated.
- Exact all-step protection preserved identity (`sim_target=0.494153`, only `-0.000244` versus A4) but reduced GarmentSim further to `0.866307`. It recovered `-66.49%` of the A4-to-B2-cont gap; the B2-cont deterioration test gave `p=2.69e-5`.
- The analysis-selected `tau=[0.0,0.2]` protocol was promoted to a co-primary full 400-image run. It reached GarmentSim `0.877685`, recovering `-0.68%` of the gap, while `sim_target=0.494381` dropped only `0.000015` from A4. Its B2-cont deterioration remained significant (`p=0.0009546`). Pose, head-pose, face detection, and detector-confidence constraints passed.
- Weak identity scale `0.3` remains a fixed 100-image secondary ablation (`GarmentSim=0.8708`) and is not used for either co-primary verdict.
- Both inference-protection verdicts are `NULL`: all-step protection worsens the garment trade-off, while late-window protection is nearly identity/garment neutral relative to A4 but recovers none of the B2-cont gap. The regression is primarily encoded in the trained A4 weights rather than removable online identity conditioning. No checkpoint was trained or modified, and this post-hoc result does not alter the frozen A4 `MIXED` verdict.
- Trainable-weight interpolation generated and evaluated 400 images at each of `alpha={0.25,0.50,0.75}` with no failures. All three middle points lie above the endpoint metric line, but the strict preregistered monotonicity test fails because `sim_target` changes from `0.494553` at `alpha=0.75` to `0.494396` at A4. The fixed verdict is `NON-LINEAR`; no post-hoc tolerance is applied.
- The `alpha=0.75` point retains GarmentSim `0.882967` versus A4's `0.877802` while slightly exceeding A4 `sim_target`. It is useful diagnostically, but does not satisfy the registered operating-point rule because it spends more than 50% of the endpoint garment cost.
- Identity-condition interpolation completed 20 paired A4/B2-cont paths (200 images) with no failures and no PuLID slerp fallback. A4 improved path efficiency (`0.7410` vs `0.7046`, `p=0.048654`) but not monotonic-violation rate significantly (`0.0750` vs `0.0813`, `p=0.483258`), so the fixed verdict is `EQUAL`, not `DECOUPLED`.
- Normalized garment drift uses exactly `max DINO drift / abs(sim_to_j(t=1)-sim_to_j(t=0))`. Its A4/B2-cont means are `0.1524/1.2677` and medians are `0.0678/0.0926`; the B2-cont mean is inflated by paths with near-zero identity denominator, which are retained rather than clipped.
- The train-only identity bank lacked all frozen test identities. Selection therefore used an eval-only 18-ID extension produced by the same F_train Glint360K recognizer (hash `4ab1d6435d639628`); held-out AdaFace was used only by the metric runner and never for pair selection.

Per preregistration, **do not launch a third mechanism-rescue run from these experiments**. The defensible project conclusion is: identity-directed counterfactual training improves identity control, but the tested objective trades away garment stability. Any future training must be framed as a new, separately preregistered study rather than an A4 retry.

### Result And Checkpoint Map

Compact reports committed to Git:

```text
docs/results/a2_gate/diagnosis.md
docs/results/a2_gate/diagnosis.json
docs/results/a4_gate/gate_report.md
docs/results/a4_gate/gate_report.json
docs/results/a4_gate/metric_report.md
docs/results/a4_gate/metric_bundle.json
docs/results/a4_gate/treatment_strength.png
```

Large local artifacts under `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2`:

```text
phase1/phase1_warmup_b2p_pulid_gatefix_resume_768x1024/   B2' run/checkpoints, 1.6G
phase1/phase2_b2_cont_r16_4000_768x1024/                  B2-cont run/checkpoints, 1.7G
phase1/phase2_a2_diff_r16_4000_768x1024/                  A2 run/checkpoints, 1.8G
phase1/phase2_a4_directed_r16_4000_768x1024/              A4 run/checkpoints, 1.9G
eval/b2p_gatefix_gen/                                     B2' 400 images, 252M
eval/b2cont_gen/                                          B2-cont 400 images, 249M
eval/a2_gen/                                              A2 400 images, 242M
eval/a4_gen/                                              A4 400 images, 259M
eval/b2p_gatefix_metrics/                                 frozen B2' metrics
eval/b2cont_metrics/                                      frozen B2-cont metrics
eval/a2_metrics/                                          frozen A2 metrics
eval/a4_metrics/                                          frozen A4 metrics
eval/id_velocity/                                         20-sample velocity tensors, CSVs, plots, report
eval/a4_prot_gen/                                         protected main run, 400 images
eval/a4_prot_metrics/                                     protected main official metrics
eval/a4_prot_tau_gen/                                     tau-window co-primary, 400 images
eval/a4_prot_tau_metrics/                                 tau-window co-primary official metrics
eval/a4_prot_scale03_gen/                                 weak-scale ablation, 100 images
eval/a4_prot_scale03_metrics/                             weak-scale official metrics
eval/a4_prot_gate_report.md                               inference-protection fixed-rule report
eval/a4_prot_gate_report.json                             machine-readable report
eval/winterp_gen/alpha{025,050,075}/                      weight interpolation, 3 x 400 images
eval/winterp_metrics/alpha{025,050,075}/                  three official metric suites per alpha
eval/weight_interp_report.md                              five-point Pareto report
eval/weight_interp_report.json                            machine-readable weight report
eval/weight_interp_pareto.png                             endpoint-line Pareto plot
eval/idinterp_gen/{a4,b2cont}/                            identity paths, 2 x 100 images
eval/idinterp_metrics/                                    selection provenance, CSVs, strips, plots
eval/identity_interp_report.{md,json}                     paired identity-path report
eval/cf_subset.json                                       shared immutable evaluation subset
```

The final trainable checkpoint for each run is under its `checkpoints/final/` directory. The A4 final checkpoint corresponds to global step 8400: B2' step 4400 plus the preregistered 4000-step continuation. The final result assets were first published in Git commit `59b1b57`.

### Resume Checklist

1. Run `git pull` and read `docs/results/a4_gate/gate_report.md` before changing training code.
2. Confirm the large local paths above still exist. Back them up before storage cleanup; GitHub does not contain checkpoints or generated images.
3. Confirm `phase1/phase2_a4_directed_r16_4000_768x1024/checkpoints/final/READY` exists. `scripts/run_a4_gate.sh` intentionally refuses a second A4 mechanism run.
4. Treat `eval/cf_subset.json`, held-out AdaFace hash `f2eb07d03de0`, DINOv2 hash `0b8b82f85de9`, and head-pose runner hash `61c34e877989` as frozen evaluation protocol state.
5. For writing/analysis, use the committed A2/A4 reports and the frozen CSVs in `eval/*_metrics/`. Do not recompute only one side of a comparison with changed weights or preprocessing.
6. Read `eval/id_velocity/analysis_report.md` and `eval/a4_prot_gate_report.md`. The protected-sampling result is diagnostic only and must not be presented as changing the frozen A4 verdict.
7. Read `eval/weight_interp_report.md` and `eval/identity_interp_report.md` before proposing another inference intervention. Their fixed verdicts are `NON-LINEAR` and `EQUAL`; neither changes the A4 `MIXED` verdict.
8. If research resumes, begin with a written new hypothesis and preregistered comparator. The current A2/A4 mechanism sequence is closed; no post-hoc lambda tuning should be reported as the same experiment.

## Clean role-flow evaluation

Build one frozen evaluation manifest from the person-disjoint clean split. The
manifest contains val and test pairs, absolute read-only human/mannequin source
paths, source SHA-256 values, source keys, and person-cluster provenance. It
uses the existing `role_test_subset.json` for test pairs when available and a
deterministic sorted fallback otherwise.

```bash
python tools/build_role_flow_eval_manifest.py \
  --dataset-root /data/muxiangyu/datasets/M2HImage/M2H_Final_v2_clean_v1 \
  --experiment-root /data/muxiangyu/experiments/M2H_Final_v2_clean_v1 \
  --output /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval_manifest.json
```

Generate images with the same manifest for A, B, and C. `--dry-run` checks
source images and cache coverage without loading FLUX.

```bash
python tools/generate_role_flow_eval.py \
  --checkpoint /path/to/checkpoints/final \
  --config configs/role_selective/A_timestep_routed.yaml \
  --manifest /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval_manifest.json \
  --output-dir /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval/A_timestep_routed/generated_final \
  --device cuda:0
```

Use `tools/evaluate_role_flow.py` with the same manifest and `--split test`
for final held-out metrics; use `--calibration-split val` for threshold
calibration. The generator does not modify the original dataset.

## Critical Notes

- Fixed on 2026-07-07: custom FLUX training/inference paths pass timestep `tau` in `[0,1]` to `FluxTransformer2DModel` and `FluxControlNetModel`. Diffusers internally multiplies by 1000.
- Checkpoints trained before this fix used corrupted time conditioning (`tau * 1e6` effective timestep) and are not reusable.
- Fixed on 2026-07-09: identity injection is pretrained PuLID-FLUX v0.9.1 only. The random projection identity route has been removed; there is no fallback or placeholder identity adapter.
- Fixed on 2026-07-10: condition gates are applied after LayerNorm, remain FP32, and use a separate 10x learning-rate group with zero weight decay. The previous ordering made scalar gates scale-invariant, so they never learned.
- Watcher failures now create a `STOP_TRAINING` marker consumed synchronously by every DDP rank. Training saves a final checkpoint and exits; watcher exits after processing `final`.
- Dataset native resolution probe found `images/human` and `images/mannequin` first 100 samples are all `768x1024`. Phase 1 cache, training, watcher, B2 generation, and mask projection now use `width=768,height=1024`.
- The obsolete 512 cache/results were deleted; active cache output is `phase1/cache_768x1024`.
- A2 is judged only against the equal-step `B2-cont` continuation from the same B2' checkpoint. B2' is a reference column, not the mechanism decision comparator.
- A2 has no canvas perturbation, VAE decode, or identity loss. Regional adaptation is implemented only by packed-token loss masks; directional identity contrast remains reserved for A4.
- A2 failed the preregistered garment axis, but `diagnose_a2.py` found the differential losses `BOUND` and held-out DeltaID gain significant (`+0.011327`, greater-side Wilcoxon `p=2.6466e-6`). This is the fixed evidence required to proceed to A4.
- A4 is a single final mechanism run. It adds semi-hard j/k sampling and a differentiable identity-directed decode loss, starts from the same B2' checkpoint as A2/B2-cont, and reuses the existing B2-cont as control. No third rescue training run is permitted.
- The completed A4 gate verdict is `MIXED`: held-out identity improved strongly (`sim_target +0.0721`, greater-side Wilcoxon `p<1e-8`), while GarmentSim regressed from `0.8951` to `0.8778` (`p=0.0010`). Per preregistration, this is reported as an identity-garment trade-off and no further mechanism run is authorized.
- The inference-only protected sampler uses two identity branches with one shared ControlNet result per Euler step: `v = v_on - M_protect * (v_on - v_off)`. Both completed co-primary fixed-rule verdicts are `NULL`; the late window preserves A4 identity and garment values but does not recover the B2-cont garment gap.
- Read-only interpolation of the 532 LoRA and 11 adapter tensors passed exact key/shape/dtype alignment and shared-start-hash checks. The strict weight-space verdict is `NON-LINEAR`, although every middle point is above the endpoint metric line.
- Identity interpolation pair selection uses only F_train Glint360K embeddings. Held-out AdaFace remains isolated in `metrics/identity_interp_metrics.py` and is never imported by generation or selection code.
- Held-out AdaFace IR-101 is evaluation-only. A4 training uses frozen Glint360K ArcFace `glintr100.onnx`, converted to a differentiable PyTorch graph with `onnx2torch`; training code fails if an AdaFace path is configured.

## Active Files

```text
configs/warmup.yaml              FLUX Phase 1 PuLID/native-resolution config
configs/a2_diff.yaml             A2: equal-step continuation with teach/invariance/hinge losses
configs/b2_cont.yaml             B2-cont: equal-step paired-only continuation
configs/a4_directed.yaml         A4: A2 losses + semi-hard sampling + directed identity decode loss
configs/a4_protected.yaml        inference-only velocity/protection protocol and fixed thresholds
configs/interpolation.yaml       pure-inference weight/identity interpolation protocol and thresholds
configs/qualitative.yaml         fixed qualitative-panel and blind human-evaluation protocol
pulid_flux.py                    frozen PuLID-FLUX v0.9.1 loader, ID embedder, transformer hook self-check
build_cache.py                   offline latent/text/PuLID-ID/appearance/garment_grid/head-pose cache
build_region_masks_z.py          CPU builder for cloth/body-bg/face packed-token masks
build_identity_bank.py           resumable ArcFace/attribute bank builder
build_identity_bank_v2.py        4-GPU Glint360K ArcFace bank used by all A4 training-side identity math
train_recognizer.py              frozen F_train loader, no-grad RetinaFace geometry, differentiable 5-point alignment
diagnose_a2.py                   A2 binding/DeltaID/tail diagnosis and fixed proceed/stop decision
train_paired.py                  paired, A2 differential, and A4 directed training; 3-card DDP by default
eval_watcher.py                  checkpoint watcher with paired and identity-swap panels
eval_b2.py                       frozen B2 subset/generation/report entry
eval_b2_metrics.py               official offline B2 metrics: held-out DeltaID, head-pose MAE, GarmentSim
eval_gate_report.py              A2 vs B2-cont fairness check, paired tests, tail analysis, verdict
eval_a4_gate_report.py           one-shot identity-axis A4 vs B2-cont preregistered verdict
analyze_id_velocity.py           real-trajectory regional energy, low-rank, and consistency probe
sampling_protected.py            isolated dual-forward protected Euler sampler and 4-GPU sharding
eval_protected_gate.py           protected vs A4/B2-cont fixed-rule paired report
interp_common.py                 shared checkpoint validation, model reuse, sharding, slerp, and status IO
interp_weights.py                aligned trainable-state interpolation and full subset generation
interp_identity.py               F_train-selected PuLID/appearance identity-path generation
metrics/identity_interp_metrics.py fixed-window LPIPS, held-out monotonicity, masked-DINO path metrics
eval_interp_reports.py           fixed weight-Pareto and identity-decoupling verdicts
qual_eval_common.py              CPU-only selection, metric CSV, path validation, and manifest helpers
make_qual_panels.py              main/A2 appendix panels, cloth zoom heatmaps, overview, and index
human_eval_build.py              balanced 240-question bank and eight self-contained blind HTML sheets
human_eval_score.py              attention QC, medians, CIs, Wilcoxon, ordinal alpha, and fixed verdicts
run_interpolation.py             shared `--part {weights,identity,all}` stage orchestrator
scripts/run_interpolation.sh     shell entry for smoke, generation, metrics, and reports
scripts/sanity_flux_timestep.py  prompt-only FLUX timestep sanity check
scripts/verify_condition_gates.py real FLUX/ControlNet/PuLID one-step gate verification
scripts/a2_vram_probe.py         real 1x ControlNet + 3x transformer differential VRAM probe
scripts/a4_vram_probe.py         complete A4 step probe including in-graph decode and F_train backward
scripts/run_a2_gate.sh           sequential A2/B2-cont training, generation, metrics, gate report
scripts/run_a4_gate.sh           unique A4 train, frozen metrics, and final PASS/FAIL/MIXED report
scripts/run_protected_inference.sh Part A, smoke, main/ablation generation, metrics, and NULL gate
scripts/run_phase1_pipeline.sh   cache check + complete gatefix pipeline
scripts/run_gatefix_to_b2.sh     4400-step train, watcher hard gate, B2' generation and metrics
scripts/run_b2_generation.sh     multi-GPU B2' generation helper
```

## Inference-only Identity Velocity And Garment Protection

This study is isolated from the frozen A4 path. It loads the final A4 checkpoint read-only and never resumes training. At each Euler step, the protected sampler computes one identity-independent ControlNet result and reuses it for the identity-on and identity-weak transformer branches:

```text
v_protected = v_on - M_protect * (v_on - v_weak)
```

Both co-primary runs use `cloth_safe`, one-token dilation, and `weak_mode=off`. One protects all timesteps; Part A selected `tau=[0.0,0.2]` for the second. The secondary 100-image ablation uses `weak_mode=scale03`. Run the stages independently:

```bash
scripts/run_protected_inference.sh analyze
scripts/run_protected_inference.sh smoke
scripts/run_protected_inference.sh tau-smoke
scripts/run_protected_inference.sh main
scripts/run_protected_inference.sh main-metrics
scripts/run_protected_inference.sh tau-main
scripts/run_protected_inference.sh tau-metrics
scripts/run_protected_inference.sh scale03-ablation
scripts/run_protected_inference.sh scale03-metrics
scripts/run_protected_inference.sh gate
```

The analysis completed 20 stratified mids and stored only that subset's bf16 velocities. Both co-primary protocols completed 400/400 images; scale03 completed 100/100, with no per-image failures. All generated images are native `768x1024`. The final report judges each co-primary independently in `eval/a4_prot_gate_report.md`; both fixed verdicts are `NULL`.

## Inference-only Weight And Identity Interpolation

These diagnostics load the frozen B2-cont and A4 endpoints read-only. They do not train, rewrite checkpoints, or alter the established A4/B2-cont generation and metric directories. Generation is deterministic at the frozen subset seeds and native `768x1024` resolution.

### Trainable-weight scan

Only the exactly aligned LoRA A/B tensors, condition projections, and gate scalars are interpolated. The base transformer, ControlNet, PuLID, and VAE remain frozen and shared.

| alpha | sim_target | DeltaID | GarmentSim |
|---:|---:|---:|---:|
| 0.00 | 0.422300 | 0.408815 | 0.895092 |
| 0.25 | 0.457956 | 0.441412 | 0.891119 |
| 0.50 | 0.482293 | 0.467024 | 0.883466 |
| 0.75 | 0.494553 | 0.479345 | 0.882967 |
| 1.00 | 0.494396 | 0.479407 | 0.877802 |

The strict verdict is `NON-LINEAR`: GarmentSim decreases monotonically, realism stays healthy, and all middle points are favorable relative to the endpoint metric line, but `sim_target` is not strictly monotonic at the final `0.75 -> 1.00` segment.

### Identity-condition paths

Twenty garment-stratified mids use the bank-v2-farthest identity pair, fixed noise, five interpolation values, and paired A4/B2-cont generation. PuLID tokens use tokenwise slerp with linearly interpolated norm; appearance uses lerp.

| path metric | A4 | B2-cont | paired p |
|---|---:|---:|---:|
| LPIPS path efficiency | 0.740980 | 0.704588 | 0.048654 |
| monotonic violation rate | 0.075000 | 0.081250 | 0.483258 |
| cloth DINO pairwise similarity | 0.975836 | 0.971422 | 0.928547 |
| sim-to-j-normalized garment drift | 0.152384 | 1.267685 | 0.985212 |

The strict verdict is `EQUAL`: efficiency alone improves significantly; held-out identity monotonicity does not. The normalized-drift means are heavy-tailed because the registered denominator can approach zero, so consult the report's medians and per-path CSV before interpreting the means.

### Reproduction

Run stages independently; `scripts/run_interpolation.sh` is a thin wrapper around the same entry point.

```bash
python run_interpolation.py --config configs/interpolation.yaml --part weights --stage generate --smoke --overwrite
python run_interpolation.py --config configs/interpolation.yaml --part weights --stage generate
python run_interpolation.py --config configs/interpolation.yaml --part weights --stage metrics
python run_interpolation.py --config configs/interpolation.yaml --part weights --stage report
python run_interpolation.py --config configs/interpolation.yaml --part identity --stage generate --smoke --overwrite
python run_interpolation.py --config configs/interpolation.yaml --part identity --stage generate
python run_interpolation.py --config configs/interpolation.yaml --part identity --stage metrics
python run_interpolation.py --config configs/interpolation.yaml --part identity --stage report
```

Full generation completed 1200/1200 weight-scan images and 200/200 identity-path images without failures. Reports are `eval/weight_interp_report.md` and `eval/identity_interp_report.md` on the local data volume.

## Qualitative Panels And Blind Human Evaluation

This final post-hoc readout is CPU-only. It reads the four frozen generation directories and existing metric CSVs; it never imports a model, generates an image, trains, or recomputes feature metrics.

The deterministic 12-mid garment-stratified selection overlaps two global extremes: `46129` is already a stratified mid and global worst-garment, while `08054` is already stratified and global best-identity. The registered rankings therefore continue without image inspection to add `41344` (worst-garment fill) and `34224` (best-identity fill), preserving exactly 16 unique mids. The global worst garment mid is `21603` with A4-B2-cont GarmentSim `-0.1118`.

Generated qualitative assets under the local data root:

```text
qual/panel_{mid}.png             16 main B2-cont / alpha=.75 / A4 panels
qual/appendix_panel_{mid}.png    same panels with an A2 appendix row
qual/cloth_zoom_{mid}.png        cloth-safe crops and 18 shared-scale difference maps
qual/overview_grid.png           16 x 3 representative comparison
qual/panel_index.{md,json}       fixed selection, metrics, tags, and paths
```

The blind bank contains exactly 240 unique normal questions: Q1=48, Q2=96, Q3=96. Every normal question is assigned to exactly three of eight raters; every rater receives 90 normal questions plus six hidden attention checks. The eight HTML files are self-contained 6-8 MB files, use opaque asset/question IDs, and contain no run, mid, jid, or filesystem path metadata. The private key hash at build time is `c72670d9a2b7ee0a`.

Run and distribute in this order:

```bash
python make_qual_panels.py --config configs/qualitative.yaml --overwrite
python human_eval_build.py --config configs/qualitative.yaml --overwrite
# Distribute only human_eval/sheets/rater_XX.html; keep key.json private.
# Put all returned CSVs under human_eval/responses/.
python human_eval_score.py --config configs/qualitative.yaml
```

The scorer excludes any rater who fails at least one of six checks. If exclusion leaves any normal question below three retained ratings, it writes `human_eval/reassignment_needed.csv` and refuses a final verdict until replacement ratings are collected. The final report will be `human_eval/report.md`; it is intentionally absent before real CSVs are returned.

For the paper body, use `qual/overview_grid.png`, `qual/panel_21603.png`, and `qual/cloth_zoom_21603.png`. Put the remaining main panels and A2 appendix rows in supplementary material.

## Spatial-condition Repair Study (2026-08-17)

This is a new study after the frozen A4 verdict. It does not rewrite the A2/A4 preregistered result. The hypothesis is that garment instance detail, head direction, and hair appearance need spatial or dense condition routes instead of low-capacity pooled semantic tokens.

The zero-shot two-control probe is complete. It reused one InstantX Union ControlNet for pose mode 4 and tile mode 1, soft-masked tile residuals to the garment token region, and evaluated the first 20 frozen mids x two identities x seed 0.

| setting | Garment-DINO to mannequin | gain vs A4 | body pose | head-5 | face detection |
|---|---:|---:|---:|---:|---:|
| A4 | 0.7607 | 0.0000 | 0.0129 | 0.0292 | 100% |
| tile 0.4 | 0.8451 | +0.0844 | 0.0111 | 0.0291 | 100% |
| tile 0.6 | 0.8765 | +0.1158 | 0.0102 | 0.0289 | 100% |
| tile 0.8 | 0.8784 | +0.1176 | 0.0102 | 0.0288 | 100% |

The registered decision is `CONFIRMED`: the best garment gain is `+0.1176`, above the `+0.05` gate, with no pose/head/face regression. The full report is `eval/tile_probe/report.md` on the data volume.

The repaired training path in `configs/spatial_warmup.yaml` therefore uses:

- `[3072 image tokens | 3072 masked-garment VAE reference tokens]`, with reference x IDs offset by 64;
- image-only pose ControlNet residuals, explicitly zero-padded over reference tokens;
- with-head mannequin pose controls, with deterministic 50/50 real-vs-synthesized head landmarks during training;
- face+hair-only gray-composited appearance crops and at most 64 dense DINOv2 hair tokens;
- legacy 64-token garment CLIP conditioning disabled, while the old path remains config-reproducible;
- optional differentiable Hair-DINO decode loss disabled in paired Phase 1 and available for directed Phase 2.

The full 6144-token rank-16 path passed the A6000 gate at `34.13 GiB` peak and `5.89 s` per micro-step, so reference stride remains 1 and LoRA rank remains 16. See `phase1/vram_report_spatial_768x1024.md`.

Run in this order:

```bash
# Step 0 is already complete; rerun only when auditing the probe.
python tile_multicontrol_probe.py --config configs/tile_probe.yaml --stage all --device cuda:0

# Four independent cache shards are inferred automatically from torchrun ranks.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
  build_cache.py --config configs/spatial_warmup.yaml --keys spatial --overwrite

# GPU0-2 train; GPU3 watches checkpoints. Training hard-pauses at run step 500.
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 \
  train_paired.py --config configs/spatial_warmup.yaml
CUDA_VISIBLE_DEVICES=3 python eval_watcher.py --config configs/spatial_warmup.yaml \
  --ckpt-dir /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_conditions_r16_4400_768x1024/checkpoints --device cuda:0

# Run only after a person has checked garment, hair, and head in the step-500 panels.
python eval_watcher.py --config configs/spatial_warmup.yaml \
  --approve-manual-review --reviewer <name>
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 \
  train_paired.py --config configs/spatial_warmup.yaml \
  --resume /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_conditions_r16_4400_768x1024/checkpoints/step-000500
# The existing GPU3 watcher keeps polling and will process later checkpoints.

# Repaired system metrics and paired repair-before comparison.
python eval_metrics_v2.py --config configs/spatial_metrics_v2.yaml --run spatial \
  --compare spatial a4 --metrics all --device cuda:0 --pose-device cuda:1
```

The manual approval file is bound to the experiment ID, review step, reviewer, and all three checklist booleans. An automatic face/swap/gate failure cannot be cleared by the approval command.

### Step-500 Hair/Region Continuation (2026-08-18)

The first spatial run remains frozen at `step-000500`. Human review found correct head direction and healthy identity response, but only coarse garment and hair resemblance. The read-only response probe showed that garment-reference conditioning was connected but spatially uniform, while the hair route was weak. Continuation config `configs/spatial_warmup_resume_hair.yaml` makes only these training changes:

- paired flow MSE uses mean-normalized token weights with `w_cloth=2.0` and `w_hair=2.0`;
- every third optimizer step, when tau is in `[0.35, 0.70]`, the paired x0 estimate may receive frozen-DINO masked hair supervision with `lambda_hair=0.1`; hair area below 1% is skipped and counted;
- appearance, hair, and head-pose gates stay in their separate fp32 optimizer group at exactly `10x` the main LR. The step-500 optimizer already had this ratio; continuation now verifies it after checkpoint restore and fails fast if it changes.

`hair_z` was added incrementally to `derived/region_masks_z` for train/val/test. Main target, pose, PuLID, appearance, garment-reference, and text caches were not rebuilt. A 20-step GPU0 smoke run completed with the full VAE-DINO graph, weighted MSE, and resume state at `42.05 GiB` peak; the first formal 3-card sparse-hair step reached `42.21 GiB`. See `docs/results/spatial_hair_region_resume_vram.md`.

The frozen step-500 trajectory baseline is:

| reading | step 500 |
|---|---:|
| cloth response concentration (energy share / area share) | 1.0333 |
| hair response / complete identity-swap response | 0.4793x |
| Garment-DINO to mannequin, fixed 10 | 0.6804 |
| Hair-DINO to reference, fixed 10 | 0.6377 |
| head-five-point distance, fixed 10 | 0.01008 |
| face detection | 100% |
| swap ArcFace cosine | 0.4930 / 0.2356 |

Artifacts are in `artifacts/response_track/step500.json` and `artifacts/response_track/response_track_curves.png`. Training-side hair loss and watcher Hair-DINO use independent code paths but the same DINOv2 checkpoint; reports disclose this limitation and retain human review plus LAB color distance as corroboration.

The watcher no longer requires final visual fidelity at steps 1000/1500. It requires rising cloth concentration and hair response, positive Garment/Hair-DINO slopes, and healthy face/swap guards. Step 2000 is the hard plateau gate. If cloth concentration remains approximately 1, hair response remains below 0.5x, or either DINO curve is flat, it writes `STOP_TRAINING` with garment-attention and hair in-context recommendations.

```bash
# Rebuild only token masks when auditing the derived asset.
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python build_region_masks_z.py \
  --config configs/spatial_warmup_resume_hair.yaml --split train,val,test --workers 24

# Required smoke test (completed).
CUDA_VISIBLE_DEVICES=0 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python train_paired.py \
  --config configs/spatial_warmup_resume_hair.yaml --dev-single-gpu \
  --allow-partial-cache --smoke-steps 20 --override-output-id spatial_hair_region_smoke20_v2

# Formal continuation and checkpoint watcher.
CUDA_VISIBLE_DEVICES=0,1,2 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=3 train_paired.py \
  --config configs/spatial_warmup_resume_hair.yaml
CUDA_VISIBLE_DEVICES=3 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python eval_watcher.py \
  --config configs/spatial_warmup_resume_hair.yaml \
  --ckpt-dir /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_region_resume_r16_4400_768x1024/checkpoints \
  --device cuda:0
```

Live tmux sessions are `m2h_spatial_hair_train` and `m2h_spatial_hair_watcher`. At step 1000, read these first: cloth concentration, hair relative response, and the two DINO slopes.

Step 1000 completed on 2026-08-18. Hair conditioning and both fixed-sample
image metrics improved, while garment-reference response had not yet become
spatially concentrated:

| reading | step 500 | step 1000 | change / slope per 500 |
|---|---:|---:|---:|
| cloth response concentration | 1.0333 | 1.0227 | -0.0107 |
| hair response / identity response | 0.4793x | 0.5697x | +0.0904x |
| Garment-DINO to mannequin | 0.6804 | 0.8214 | +0.1410 |
| Hair-DINO to reference | 0.6377 | 0.7646 | +0.1269 |
| face detection | 100% | 100% | healthy |
| maximum swap cosine | 0.4930 | 0.3265 | healthy |

The registered status is `trend_not_yet_approved`: no step-1000 approval file
was written because cloth concentration did not rise. Training continues to the
step-1500 trend check; step 2000 remains the only hard plateau stop.
The exact snapshot is `artifacts/response_track/step1000.json`.

After step 4400 passes the trajectory gate, generate the frozen subset into the continuation-specific directory and run the full v2 comparison without overwriting the earlier spatial run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=4 eval_b2.py \
  --config configs/spatial_warmup_resume_hair.yaml \
  --ckpt /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_region_resume_r16_4400_768x1024/checkpoints/final \
  --subset /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/eval/cf_subset.json
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python eval_metrics_v2.py \
  --config configs/spatial_metrics_v2.yaml --run spatial_hair_region \
  --compare spatial_hair_region a4 --metrics all --device cuda:0 --pose-device cuda:1
```

The same sequence is queued by `scripts/run_spatial_hair_finalize.sh`. It waits without using a GPU and refuses to generate if the trajectory watcher writes `STOP_TRAINING`.

### Step-2000 Hair In-Context Continuation (2026-08-18)

The step-2000 review accepted the garment and head routes: fixed-sample
Garment-DINO reached `0.916`, head-five-point distance reached `0.012`, and
the garment curve was still rising. The only unresolved route was hair. The
read-only gate inspection is recorded in
`docs/results/hair_gate_step2000_diagnosis.md` and concluded `INCONCLUSIVE`:
the gate is in the `condition_gates_fp32` optimizer group at exactly `10x`
LR, its real-batch gradient is finite and nonzero, the projected token slice is
present after LayerNorm, but the checkpoint value remains `0.100031823`.

The continuation config is
`configs/spatial_warmup_resume_hair_incontext.yaml`. It keeps the garment and
head paths unchanged and adds a hair-only in-context latent segment:

```text
[image 3072 | garment reference 3072 | hair reference 3072]
```

The hair reference contains only FASHN label 2 at native `768x1024`; all other
pixels are neutral gray. It is VAE-encoded into `hair_ref_latents`, uses a
separate `y+64` image-ID offset, receives no ControlNet residual, and does not
participate in flow loss. The old hair-token parameters remain in the graph so
the step-2000 Adam state restores exactly, but their condition signal is off by
default. Hair-DINO supervision now uses `lambda_hair=0.5` every second eligible
optimizer step in the tau window `[0.35, 0.70]`.

The complete 9216-token step, including differentiable VAE decode, frozen DINO,
backward, and optimizer update, peaks at `43.634 GiB` and takes `10.03 s` on an
A6000. It passes the 44 GiB gate, so both garment and hair references remain at
full token resolution. See
`docs/results/vram_report_hair_incontext_768x1024.md`.
The deliberately conservative all-heavy-step bound is `8.36 h` from step 2000
to 2500 and `40.12 h` to step 4400; the first 20 formal optimizer steps write a
mixed-workload estimate to `logs/benchmark.json`.
The required 20-step smoke completed from step 2000 to 2020 at `43.712 GiB`,
`10.102 s/step`, with the Hair-DINO decode path active and a `10%` skip rate.

Run the incremental cache upgrade and smoke test before formal continuation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=4 build_cache.py \
  --config configs/spatial_warmup_resume_hair_incontext.yaml \
  --split train,val,test --keys hair_ref_latents,hair_ref_empty

CUDA_VISIBLE_DEVICES=0 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python \
  train_paired.py --config configs/spatial_warmup_resume_hair_incontext.yaml \
  --dev-single-gpu --allow-partial-cache --smoke-steps 20 \
  --override-output-id spatial_hair_incontext_smoke20
```

After the smoke succeeds, launch GPU0-2 training and the GPU3 watcher together:

```bash
bash scripts/run_spatial_hair_incontext_resume.sh
```

On its first launch, the script recomputes step 2000 with the new route and the
same ten visible-hair validation samples used by later checkpoints. This avoids
turning the old `6 valid / 4 no_hair` sample mix into a false Hair-DINO or LAB
slope. The one-off baseline uses GPU3 while continuation training starts on
GPU0-2; the persistent checkpoint watcher takes GPU3 after baseline completion.
Because the appended hair segment is untrained at this point, step 2000 records
its zero-step impact but does not fire post-training garment/head guards. Those
absolute guards begin at step 2500; the audited first attempt that applied them
to the baseline is retained under the non-`v2` run ID.

The final evaluation can wait in a separate terminal; it refuses to run after
any watcher stop and writes only continuation-specific generation/metric paths:

```bash
bash scripts/run_spatial_hair_incontext_finalize.sh
```

The corrected trajectory gate treats garment response concentration as a
descriptive probe only. Garment stopping now follows Garment-DINO decline or a
human `VISUAL_REGRESSION` marker. At step 2500, inspect Hair-DINO slope,
Hair-LAB distance slope, and `hair_ref_swap_concentration` first. Step 3000 is
the hair hard gate: it stops only when Hair-DINO is flat and LAB distance is not
falling. Garment-DINO `>=0.90`, head-five-point distance `<=0.02`, 100% face
detection, and the baseline-bound identity-swap cosine remain immediate guards.

## PuLID Assets

Required paths are configured under `model.pulid` in `configs/warmup.yaml`:

```text
repo: /data/muxiangyu/modelLibrary/PuLID
weight_path: /data/muxiangyu/modelLibrary/PuLID/models/pulid_flux_v0.9.1.safetensors
antelopev2_dir: /data/muxiangyu/modelLibrary/PuLID/models/antelopev2
hf_home: /data/muxiangyu/modelLibrary
```

Current PuLID-FLUX weight hash prefix: `92c41c3af322b02e`. Startup fails if these assets are missing. The loader also runs two self-checks: PuLID CA delta and real FLUX transformer output delta.

For A2, the transformer self-check additionally compares two different PuLID contexts at fixed latent/timestep. PuLID context tensors travel through the non-reentrant checkpoint graph explicitly, so i/j/k backward recomputation cannot reuse the final context accidentally.

## Phase 1 Execution Order

1. Probe/confirm native resolution and run timestep sanity if needed.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python scripts/sanity_flux_timestep.py   --base /data/muxiangyu/pythonPrograms/M2HImage/models/hf/black-forest-labs/FLUX.1-dev   --device cuda:0 --height 1024 --width 768 --steps 20   --out-dir /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/timestep_sanity
```

2. Run the native-resolution VRAM stress test and use the adopted config in `vram_report_768x1024.md`.

3. Rebuild the Phase 1 cache on 4 GPUs. This writes `target_latents/pose_latents` at `(3072,64)`, `pulid_id_embed` at `(32,2048)`, `garment_grid`, 1.8x head-crop `appearance`, raw `head_pose`, and debug crops under `phase1/cache_768x1024/debug_head_crops/`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -m torch.distributed.run   --nproc_per_node=4 build_cache.py --config configs/warmup.yaml --split train,val,test --overwrite
```

4. Verify the corrected gates on one real training step. All three gate gradients must be finite/nonzero and each optimizer update must exceed `1e-4`.

```bash
CUDA_VISIBLE_DEVICES=0 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python scripts/verify_condition_gates.py --config configs/warmup.yaml --device cuda:0
```

5. Run the corrected 4400-step (about one day) warmup on GPU0-2 with GPU3 watcher. Step 500 is a hard gate: face detection >=95%, swap cosine <0.85, and all condition gates must move from init by more than 0.001. If it passes, the same script automatically generates B2' on four GPUs and runs all official metrics.

```bash
bash scripts/run_gatefix_to_b2.sh
```

Outputs:

```text
phase1/phase1_warmup_b2p_pulid_gatefix_resume_768x1024/
eval/b2p_gatefix_gen/
eval/b2p_gatefix_metrics/
eval/b2p_gatefix_report.md
```

## Watcher Checks

`eval_watcher.py` writes five-column swap panels for the first `eval.identity_swap_count` validation samples:

```text
[m_i | pose | generated(c_i) | generated(swap c_j) | h_i]
```

Each watcher report includes face detection rate, ArcFace paired-vs-swap cosine, and the three condition-token gate values. At step 500 it writes a real `STOP_TRAINING` sentinel if face detection is below 95%, swap cosine is not below 0.85, or gates have not moved. DDP broadcasts that decision to every rank, saves `final`, and exits before B2'.

For A4, the first two panels add two counterfactual outputs generated from the same noise:

```text
[m_i | pose | generated(c_i) | generated(c_j) | generated(c_k) | h_i]
```

The report also plots training `sim_gap`, cumulative identity-loss face-detection skip rate, and emits a top-level warning if the skip rate exceeds 50%.

## Cache Schema

Per-sample `npz` files must contain:

```text
target_latents   # (3072, 64) for 768x1024
pose_latents     # (3072, 64) for 768x1024
pulid_id_embed   # (32, 2048), official PuLID-FLUX ID tokens
appearance       # 1.8x expanded head crop visual feature
garment_grid     # shape (N <= 64, dim), patch-token grid feature
head_pose        # raw token, no cache-time dropout
```

Training applies `training.head_pose_dropout` dynamically in `PairedWarmupDataset`; eval/watcher/B2 use dropout 0.

## A2 Differential Definition

For a shared paired sample latent `z_tau`, A2 computes one ControlNet result and reuses it for three transformer calls:

```text
paired: PuLID(i) + appearance(i)
CF-j:   PuLID(j) + appearance(j)
CF-k:   PuLID(k) + appearance(k)
```

Garment grid, head pose, pose ControlNet, prompt, `z_tau`, and `tau` remain from sample `i`. The losses are:

```text
L = L_pair + 0.5 L_teach + 0.2 L_inv + 0.05 L_hinge
```

Differential losses run only for `tau in [0.2,0.8]`. During the first 200 continuation steps, teach/invariance remain active while hinge weight is zero and `g` is calibrated as `Q25(face_diff / d_arc)`. The resolved value is written to `resolved_config.yaml`, `hinge_calibration.json`, logs, and checkpoints.

## A2 Additional Assets

```text
derived/region_masks_z/{id}.npz
  cloth_safe_z   # (3072,) float16
  body_bg_z      # (3072,) float16
  face_z         # (3072,) float16, source id_strong

derived/identity_bank.npz
  ids
  embeds         # (36034, 512), normalized ArcFace; sampling/calibration only
  gender
  age
  age_group
  skin_cluster
```

Identity compatibility is same gender, skin-cluster distance at most 1, and age distance at most 15. The bank embedding is never sent to FLUX; j/k model conditions still come from cached PuLID tokens and appearance features.

The exact compatibility audit found one infeasible source, `47160`, with only one eligible counterfactual identity. It is excluded as a source in both A2 and B2-cont instead of relaxing the protocol; it remains available as a donor for other compatible samples. Dataset startup validates that every active A2 source has at least two candidates. `launch.json` records the ordered train-ID hash, sample count, and exclusion list, and the gate report treats any mismatch as a fairness blocker.

## A2 Execution Order

1. Build token masks and inspect the 20 overlays under `derived/region_masks_z/debug/`.

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python build_region_masks_z.py \
  --config configs/a2_diff.yaml --split train --workers 24 --debug-count 20
```

2. Build the resumable single-file identity bank.

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python build_identity_bank.py \
  --config configs/a2_diff.yaml --workers 8
```

3. Run the real full-differential VRAM probe.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python scripts/a2_vram_probe.py \
  --config configs/a2_diff.yaml --device cuda:0
```

Measured adoption at 768x1024: rank 16, `diff_every=1`, one ControlNet plus three transformer forwards, peak `35.48 GiB`; no rank or frequency reduction is required. The report is `phase1/vram_report_diff_768x1024.md`.

4. Run the required 20-step single-GPU smoke. Its first step forces `tau=0.5`, and smoke-only `g` makes all three losses executable.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python train_paired.py \
  --config configs/a2_diff.yaml --dev-single-gpu --smoke-steps 20 \
  --override-output-id phase2_a2_smoke_20
```

5. Run A2 and B2-cont sequentially with identical B2' resume hash, sampler state, seed, global batch, LR, LoRA rank, and 4000 continuation steps; then generate and evaluate both with the frozen subset.

```bash
bash scripts/run_a2_gate.sh
```

Final outputs:

```text
eval/a2_gen/
eval/a2_metrics/
eval/b2cont_gen/
eval/b2cont_metrics/
eval/gate_garment_per_mid_hist.png
eval/gate_report.md
```

`eval_gate_report.py` blocks the verdict if fairness fields differ. It reports A2/B2-cont/B2' side by side, paired per-mannequin GarmentSim and pose-variance Wilcoxon tests, bottom-quartile GarmentSim, DeltaID regression, effect sizes, and the fixed PASS/MIXED/FAIL rule.

## A2 Diagnosis And A4 Execution Order

The committed diagnosis under `docs/results/a2_gate/diagnosis.md` is the only transition gate into A4:

```text
differential binding: BOUND
resolved hinge_g: 0.0291450452
hinge activation mean: 12.89%
held-out DeltaID gain: +0.011327
greater-side Wilcoxon p: 2.6466e-6
decision: PROCEED
```

1. Re-run the diagnosis only to verify immutable inputs. A `NOT-SIGNIFICANT` result stops A4.

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python diagnose_a2.py
```

2. Build the training-recognizer identity bank on four GPUs. Tight face crops are uniformly padded, enlarged, RetinaFace-aligned, and embedded by frozen Glint360K ArcFace. The builder is resumable and fails on any missing identity.

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -m torch.distributed.run \
  --nproc_per_node=4 build_identity_bank_v2.py --config configs/a4_directed.yaml --batch-size 64
```

Output: `derived/identity_bank_v2.npz`. This bank supplies semi-hard distances, hinge calibration, and A4 identity references. It is never sent into FLUX as a condition.

3. Probe the complete triggered A4 step. The adopted row must be at most 44 GiB.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python scripts/a4_vram_probe.py \
  --config configs/a4_directed.yaml --device cuda:0
```

Measured result: full-resolution decode peaked at `44.0168 GiB`, so the strict gate selected the documented first fallback. Half-resolution latent decode (`latent_scale=0.5`) peaked at `37.4741 GiB`; `decode_freq=3`, LoRA rank 16, and full transformer checkpointing remain unchanged. Generation and evaluation still run at native 768x1024.

4. Run the required 20-step full-branch smoke. Step 1 forces `tau=0.5`, so all three transformer forwards, one VAE decode, RetinaFace geometry, F_train, both directed losses, and joint backward execute.

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python train_paired.py \
  --config configs/a4_directed.yaml --dev-single-gpu --smoke-steps 20 \
  --override-output-id phase2_a4_smoke_20
```

5. Run the unique A4 continuation and frozen evaluation. GPU0-2 train; GPU3 watches checkpoints. The script refuses to launch a second training run once `checkpoints/final/READY` exists.

```bash
bash scripts/run_a4_gate.sh
```

Outputs:

```text
phase1/phase2_a4_directed_r16_4000_768x1024/
eval/a4_gen/
eval/a4_metrics/
eval/a4_report.md
eval/a4_gate_report.md
eval/a4_gate_report.json
```

The final identity gate requires held-out `sim_target` gain at least 0.03 with greater-side Wilcoxon `p<0.05`; GarmentSim, pose cross-identity variance, face detection, and detector-confidence realism proxy must not regress. The report emits the fixed PASS, FAIL, or MIXED conclusion and does not authorize another training round.

## Grep Disposition

- `512` remains only for non-resolution meanings such as text max length and ArcFace embedding size.
- `1000` remains for documented timestep sanity (`timestep / 1000`) and numeric constants unrelated to model timestep scaling; training/watcher/B2 pass `tau` in `[0,1]`.
- `resolution` call sites now use `get_resolution()` and pass `(width,height)` through cache, training, watcher, B2 generation, and GarmentSim mask projection.
