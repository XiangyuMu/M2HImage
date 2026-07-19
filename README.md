# M2HImage FLUX Phase 1 Warmup / B2' / A2 / A4 One-shot Gate

This repository contains the FLUX.1-dev MA-RA-CDT paired warmup, B2' adapter-only baseline, A2 differential counterfactual experiment, and the preregistered one-shot A4 identity-directed gate.

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
- Held-out AdaFace IR-101 is evaluation-only. A4 training uses frozen Glint360K ArcFace `glintr100.onnx`, converted to a differentiable PyTorch graph with `onnx2torch`; training code fails if an AdaFace path is configured.

## Active Files

```text
configs/warmup.yaml              FLUX Phase 1 PuLID/native-resolution config
configs/a2_diff.yaml             A2: equal-step continuation with teach/invariance/hinge losses
configs/b2_cont.yaml             B2-cont: equal-step paired-only continuation
configs/a4_directed.yaml         A4: A2 losses + semi-hard sampling + directed identity decode loss
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
scripts/sanity_flux_timestep.py  prompt-only FLUX timestep sanity check
scripts/verify_condition_gates.py real FLUX/ControlNet/PuLID one-step gate verification
scripts/a2_vram_probe.py         real 1x ControlNet + 3x transformer differential VRAM probe
scripts/a4_vram_probe.py         complete A4 step probe including in-graph decode and F_train backward
scripts/run_a2_gate.sh           sequential A2/B2-cont training, generation, metrics, gate report
scripts/run_a4_gate.sh           unique A4 train, frozen metrics, and final PASS/FAIL/MIXED report
scripts/run_phase1_pipeline.sh   cache check + complete gatefix pipeline
scripts/run_gatefix_to_b2.sh     4400-step train, watcher hard gate, B2' generation and metrics
scripts/run_b2_generation.sh     multi-GPU B2' generation helper
```

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
