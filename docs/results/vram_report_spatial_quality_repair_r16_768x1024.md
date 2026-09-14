# Spatial Quality Repair VRAM Probe (R16, 768x1024)

## Decision

**PASS: peak VRAM is below the 44 GiB training cap.**

The final three-rank DDP topology with the real `grad_accum=6` reached
`43.826136 GiB` allocated on rank 0. All three ranks completed the replay
without OOM. The margin is `0.173864 GiB` (about 178 MiB), so the run is
admitted but remains a tight-memory configuration.

## Probed Step

- GPU: NVIDIA RTX A6000, 48 GiB physical memory
- precision: FLUX/ControlNet/VAE BF16; frozen training-side DINO BF16
- trainable state: FLUX LoRA rank 16 plus the existing condition adapter
- sequence: 3072 image + 3072 garment-reference + 768 hair-reference tokens
- decode latent scale: 0.48 for the sparse identity/hair supervision branch
- graph: one identity-independent ControlNet call, three transformer forwards,
  one in-graph VAE decode, then either frozen DINO hair loss or frozen ArcFace
  directed identity loss, followed by the joint backward
- checkpointing: transformer, VAE decode, and DINO forward enabled
- PuLID, pack/unpack, timestep tau, and checkpoint parameters are unchanged

## Measurements

| probe | topology | steps | peak GiB | seconds/optimizer step | result |
|---|---|---:|---:|---:|---|
| initial | 1 GPU, DINO FP32, scale .50 | 20 | 44.005098 | 20.5104 | FAIL (>44) |
| mitigation 1 | 1 GPU, DINO BF16, scale .50 | 20 | 43.843829 | 20.4939 | PASS |
| incomplete topology | 3-GPU DDP, accum 1, scale .50 | 2 | 43.929776 | n/a | PASS |
| real topology | 3-GPU DDP, accum 6, scale .50 | 1 | 44.014976 | n/a | FAIL (>44) |
| final smoke | 1 GPU, DINO BF16, scale .48 | 20 | 43.654990 | 20.6357 | PASS |
| final topology | 3-GPU DDP, accum 6, scale .48 | 1 | 43.826136 | n/a | PASS |

The 20-step smoke forces `tau=0.5`, so every micro-step executes all three
transformer branches and alternates the two decode objectives. It is a
conservative per-micro-step throughput measurement. The final smoke recorded
ten hair-loss attempts, one area skip (`10%`), non-zero `mse_face`,
non-zero directed losses/`sim_gap`, one ControlNet call, and three transformer
calls on every step.

## Mitigation

The initial single-GPU result exceeded the cap by about 5 MiB, so the frozen
*training-side* DINO instance was converted from FP32 to BF16 through
`training.hair_loss.dino_precision`. A short DDP probe then appeared to pass,
but the first real `grad_accum=6` optimizer step reached `44.014976 GiB`.
That launch was stopped at step 4401 and archived as
`phase1_spatial_quality_repair_r16_6400_768x1024_rejected_vram_scale050_step4401`;
it produced no checkpoint and is not a training result.

The sparse decode latent scale was then reduced from `0.50` to `0.48`. ArcFace
and DINO still receive their fixed 112x112 and 518x518 inputs, respectively.
Replaying the same seed/sampler step with the real accumulation topology lowered
the peak by `0.188839 GiB`. Neither mitigation changes trainable checkpoint
parameters, reference token counts, output resolution, or the held-out evaluator.

No hair-token reduction, LoRA-rank reduction, decode-frequency reduction, DDP
change, or checkpoint architecture change was required.

## Runtime Estimate

The single-GPU worst-case smoke is not the formal optimizer cadence: the
three-rank run uses gradient accumulation 6, while differential branches are
active only when sampled tau lies in [0.2, 0.8] and decode losses are sparse.
A conservative launch estimate for 2000 optimizer steps is roughly 52-68 hours.
The authoritative estimate is `logs/benchmark.json` after the first 20 formal
optimizer steps; it replaces this range because it observes the real tau hit
rate, accumulation, dataloader, and DDP synchronization.

## Artifacts

- final smoke: `phase1/spatial_quality_repair_smoke20_bf16_scale048/`
- real-accumulation probe: `phase1/spatial_quality_repair_formal_probe1_scale048/`
- rejected scale-0.50 launch: `phase1/phase1_spatial_quality_repair_r16_6400_768x1024_rejected_vram_scale050_step4401/`
- resolved config:
  `configs/spatial_quality_repair_resume.yaml`

