# Spatial Conditioning VRAM Report 768x1024

- measured path: paired flow, one frozen pose ControlNet, FLUX over [image | garment-reference | hair-reference], PuLID, differentiable VAE/DINO hair loss, and joint backward/optimizer
- garment reference remains full resolution; only the hair reference may use stride 2
- ControlNet runs on image tokens only; its residual is zero-padded over both reference segments
- safety gate: torch.cuda.max_memory_allocated() <= 44 GiB on A6000 48G
- fallback order: hair stride 1 -> hair stride 2 -> full checkpointing -> rank 8

| hair stride | rank | image tokens | garment ref | hair ref | total tokens | status | failed stage | peak GiB | step sec |
|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 16 | 3072 | 3072 | 3072 | 9216 | ok |  | 43.63 | 10.03 |

Conclusion: use garment stride=1, hair stride=1, LoRA rank=16; peak=43.63 GiB.
Conservative 3-GPU 4400-step estimate: 73.57 h (probe step x grad_accum=6; data loading/watcher excluded).
For the actual step-2000 to step-4400 continuation, the same all-heavy-step upper
bound is 40.12 h; reaching step 2500 is 8.36 h. Formal training should be faster
because Hair-DINO decode runs only every second optimizer step and only inside
the configured tau window. `logs/benchmark.json` supersedes these bounds after
the first 20 resumed optimizer steps.

The required 20-step real smoke restored optimizer and sampler state from
step 2000, reached step 2020, and completed with `status=complete`. It measured
`43.712 GiB` peak and `10.102 s/optimizer-step` with `grad_accum=1`; the three
reference/image segments were all 3072 tokens, Hair-DINO decode and backward
were exercised, and the observed hair-loss skip rate was `10%`.
