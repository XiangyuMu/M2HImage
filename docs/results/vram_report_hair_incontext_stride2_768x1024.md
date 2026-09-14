# Spatial Conditioning VRAM Report 768x1024

- measured path: paired flow, one frozen pose ControlNet, FLUX over [image | garment-reference | hair-reference], PuLID, differentiable VAE/DINO hair loss, and joint backward/optimizer
- garment reference remains full resolution; only the hair reference may use stride 2
- ControlNet runs on image tokens only; its residual is zero-padded over both reference segments
- safety gate: torch.cuda.max_memory_allocated() <= 44 GiB on A6000 48G
- fallback order: hair stride 1 -> hair stride 2 -> full checkpointing -> rank 8

| hair stride | rank | image tokens | garment ref | hair ref | total tokens | status | failed stage | peak GiB | step sec |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 2 | 16 | 3072 | 3072 | 768 | 6912 | ok |  | 42.36 | 7.25 |

Conclusion: use garment stride=1, hair stride=2, LoRA rank=16; peak=42.36 GiB.
Conservative 3-GPU 4400-step estimate: 53.18 h (probe step x grad_accum=6; data loading/watcher excluded).

## Formal DDP Verification

- resume source: fixed-protocol branch from `step-002500`
- world size / accumulation / global batch: `3 / 6 / 18`
- measured first 20 optimizer steps: `848.687 s`
- measured optimizer-step time: `42.434 s`
- measured torch peak allocation: `42.607 GiB` (PASS under the 44 GiB gate)
- remaining 1900-step continuation estimate: `22.40 h`, excluding watcher and final evaluation
