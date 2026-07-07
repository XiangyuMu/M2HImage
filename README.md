# M2HImage FLUX Phase 1 Warmup / B2 Baseline

This repository contains the current FLUX.1-dev Phase 1 path for MA-RA-CDT paired flow warmup and B2 adapter-only baseline evaluation.

## Critical Notes

- Fixed on 2026-07-07: custom FLUX training/inference paths pass timestep `tau` in `[0,1]` to `FluxTransformer2DModel` and `FluxControlNetModel`. Diffusers internally multiplies by 1000.
- Checkpoints trained before this fix used corrupted time conditioning (`tau * 1e6` effective timestep) and are not reusable. Rebuild cache and retrain warmup before treating B2 numbers as the A2 baseline.
- Local search found no PuLID-FLUX or InfiniteYou implementation/weights in this environment. The current identity adapter is the explicit fallback: projection tokens with zero-init gates and adapter projection LR multiplier. This changes the A9 ablation premise until a mature FLUX identity adapter is installed.

## Active Files

```text
configs/warmup.yaml              FLUX Phase 1 config
build_cache.py                   offline latent/text/identity/appearance/garment_grid/head-pose cache
train_paired.py                  paired flow warmup training, 3-card DDP by default
eval_watcher.py                  checkpoint watcher with paired and identity-swap panels
eval_b2.py                       frozen B2 subset/generation/report entry
eval_b2_local_metrics.py         local generated-image proxy metrics
eval_b2_make_vis.py              B2 contact sheets
scripts/sanity_flux_timestep.py  Fix 1 prompt-only FLUX timestep sanity check
scripts/run_phase1_pipeline.sh   cache + training + watcher launch helper
scripts/run_b2_generation.sh     4-GPU B2 generation helper
```

## Phase 1 Execution Order

1. Run the timestep sanity check and inspect that `custom_prompt_only_tau_0_1.png` is a clean image, not residual noise:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python scripts/sanity_flux_timestep.py \
  --base /data/muxiangyu/pythonPrograms/M2HImage/models/hf/black-forest-labs/FLUX.1-dev \
  --device cuda:0 \
  --steps 20 \
  --out-dir /data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/timestep_sanity
```

2. Rebuild the Phase 1 cache on 4 GPUs. This writes `garment_grid` patch tokens, 1.8x head-crop appearance features, raw head-pose tokens, and debug crops under `phase1/cache_512/debug_head_crops/`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/muxiangyu/miniconda3/envs/refton_m2h/bin/python -m torch.distributed.run \
  --nproc_per_node=4 build_cache.py --config configs/warmup.yaml --split train,val,test --overwrite
```

3. Retrain warmup for 6000 steps with 3-card DDP plus GPU3 watcher. From step 500, inspect watcher reports for face detection collapse and identity-swap response:

```bash
bash scripts/run_phase1_pipeline.sh
```

4. Re-run B2 with the existing frozen `eval/cf_subset.json`; these post-fix numbers are the valid A2 comparison baseline:

```bash
bash scripts/run_b2_generation.sh
/home/muxiangyu/miniconda3/envs/imagdressing_m2h/bin/python eval_b2_local_metrics.py --config configs/warmup.yaml --device cuda:3
```

## Watcher Checks

`eval_watcher.py` now writes five-column swap panels for the first `eval.identity_swap_count` validation samples:

```text
[m_i | pose | generated(c_i) | generated(swap c_j) | h_i]
```

Each watcher report includes:

- face detection rate across all generated paired/swap images, with `⚠ RED: face detection collapsed` when below 90%;
- ArcFace cosine between paired and swapped generated faces, with `⚠ adapter not responding` when cosine is above 0.9.

## Cache Schema

Per-sample `npz` files are expected to contain:

```text
target_latents
pose_latents
identity
appearance
garment_grid     # shape (N <= 64, dim)
head_pose        # raw token, no cache-time dropout
```

Training applies `training.head_pose_dropout` dynamically in `PairedWarmupDataset`; eval/watcher/B2 use dropout 0.

## Validation Run Notes

Validated in this workspace:

- `scripts/sanity_flux_timestep.py` ran successfully with 20 steps. Official and custom prompt-only outputs had identical pixel stats.
- `build_cache.py --split val --limit 1 --overwrite` produced `garment_grid` shape `(64, 1024)`, a nonzero raw head-pose token, and a debug head crop.
- `eval_watcher.py --ckpt .../step-006000` produced five-column swap panels and a report with face-detection and swap checks. The old checkpoint is only for watcher structure validation and remains invalid for B2 because it was trained before the timestep fix.
