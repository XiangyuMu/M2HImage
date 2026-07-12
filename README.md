# M2HImage FLUX Phase 1 Warmup / B2' Baseline

This repository contains the current FLUX.1-dev Phase 1 path for MA-RA-CDT paired flow warmup and B2/B2' adapter-only baseline evaluation.

## Critical Notes

- Fixed on 2026-07-07: custom FLUX training/inference paths pass timestep `tau` in `[0,1]` to `FluxTransformer2DModel` and `FluxControlNetModel`. Diffusers internally multiplies by 1000.
- Checkpoints trained before this fix used corrupted time conditioning (`tau * 1e6` effective timestep) and are not reusable.
- Fixed on 2026-07-09: identity injection is pretrained PuLID-FLUX v0.9.1 only. The random projection identity route has been removed; there is no fallback or placeholder identity adapter.
- Fixed on 2026-07-10: condition gates are applied after LayerNorm, remain FP32, and use a separate 10x learning-rate group with zero weight decay. The previous ordering made scalar gates scale-invariant, so they never learned.
- Watcher failures now create a `STOP_TRAINING` marker consumed synchronously by every DDP rank. Training saves a final checkpoint and exits; watcher exits after processing `final`.
- Dataset native resolution probe found `images/human` and `images/mannequin` first 100 samples are all `768x1024`. Phase 1 cache, training, watcher, B2 generation, and mask projection now use `width=768,height=1024`.
- The obsolete 512 cache/results were deleted; active cache output is `phase1/cache_768x1024`.

## Active Files

```text
configs/warmup.yaml              FLUX Phase 1 PuLID/native-resolution config
pulid_flux.py                    frozen PuLID-FLUX v0.9.1 loader, ID embedder, transformer hook self-check
build_cache.py                   offline latent/text/PuLID-ID/appearance/garment_grid/head-pose cache
train_paired.py                  paired flow warmup training, 3-card DDP by default
eval_watcher.py                  checkpoint watcher with paired and identity-swap panels
eval_b2.py                       frozen B2 subset/generation/report entry
eval_b2_metrics.py               official offline B2 metrics: held-out DeltaID, head-pose MAE, GarmentSim
scripts/sanity_flux_timestep.py  prompt-only FLUX timestep sanity check
scripts/verify_condition_gates.py real FLUX/ControlNet/PuLID one-step gate verification
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

## Grep Disposition

- `512` remains only for non-resolution meanings such as text max length and ArcFace embedding size.
- `1000` remains for documented timestep sanity (`timestep / 1000`) and numeric constants unrelated to model timestep scaling; training/watcher/B2 pass `tau` in `[0,1]`.
- `resolution` call sites now use `get_resolution()` and pass `(width,height)` through cache, training, watcher, B2 generation, and GarmentSim mask projection.
