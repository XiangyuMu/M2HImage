# Spatial Hair/Region Continuation VRAM

Date: 2026-08-18

Protocol: FLUX.1-dev + InstantX Union ControlNet + PuLID-FLUX + rank-16 LoRA,
768x1024, 3072 image tokens plus 3072 garment-reference tokens. Paired hair
supervision uses an in-graph VAE decode and frozen DINOv2 masked-patch loss.

## Smoke Gate

- checkpoint: frozen spatial `step-000500`
- command config: `configs/spatial_warmup_resume_hair.yaml`
- optimizer steps: 20, single A6000, micro-batch 1
- smoke tau: fixed at 0.5 so the hair quality window is exercised
- peak allocated VRAM: 42.05 GiB
- mean wall time: 6.46 seconds per optimizer step
- result: PASS (`<=44 GiB`), including valid hair loss and skip paths

## Formal DDP Confirmation

- world size: 3, micro-batch 1, gradient accumulation 6, global batch 18
- first sparse-hair log: step 502
- peak allocated VRAM: 42.21 GiB
- gate group: 3 fp32 parameters at 5.625e-4 LR, exactly 10x the main LR
- result: PASS (`<=44 GiB`)

`nvidia-smi` process memory is higher because it includes CUDA reserved memory,
NCCL buffers, and context allocations. The preregistered gate uses PyTorch peak
allocated memory, which is the value logged in `train.jsonl` and above.
