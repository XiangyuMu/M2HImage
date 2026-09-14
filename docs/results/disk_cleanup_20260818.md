# M2HImage Disk Cleanup - 2026-08-18

## Scope

Only the current M2HImage project was cleaned. The user narrowed the earlier
cross-project plan before any deletion started. Mannequin2RealVideo,
StableAnimator, M2H_ablation, GSVTON, modelLibrary, and their files were not
modified.

## Result

| reading | before | after |
|---|---:|---:|
| `/data` available bytes | 98,473,025,536 | 105,024,417,792 |
| bytes released | - | 6,551,392,256 |
| released GiB | - | 6.10 |
| M2H phase1 reported size | 73G | 67G |

Deleted assets:

- Eight directories whose experiment names explicitly contained `smoke`.
- Seven early checkpoints each from A4, B2-cont, and A2 (steps 4500-7500).
- Six early checkpoints from the previous gate-fix warmup (steps 1000-3500).

## Retained Assets

- The active `phase1_spatial_hair_region_resume_r16_4400_768x1024` run and all
  of its checkpoints, logs, watcher outputs, and finalizer state.
- Both `cache_spatial_768x1024` and `cache_768x1024`.
- Every generated image, validation image, metric CSV, report, debug image,
  source dataset, and model weight.
- A4, B2-cont, and A2 each retain both `checkpoints/final` and
  `checkpoints/step-008000`.
- The previous gate-fix warmup retains both `checkpoints/final` and
  `checkpoints/step-004000`.

The final and highest numbered checkpoint hashes differed in all four runs, so
both copies were intentionally retained.

## Post-Cleanup Checks

- Current training advanced from step 1426 before cleanup to step 1432 after
  cleanup; GPU0-2 remained active at full utilization.
- The three external project sizes remained unchanged: Mannequin2RealVideo
  851G, StableAnimator 388G, and M2H_ablation 521G.
- No active cache, current checkpoint, or generated-result directory appeared
  in any deletion command.
