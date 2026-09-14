# Spatial Quality Repair: Reference Leakage, High Frequency, and Identity

## Scope

This continuation repairs three visual failures found after spatial step 4400:
garment high-frequency blur, identity/face degradation, and reference-canvas
leakage. It resumes the immutable step-4400 checkpoint and does not alter the
validated flow timestep, latent pack/unpack convention, ControlNet head control,
PuLID hook/self-check, DDP, sampler, or atomic checkpoint format.

## Part A: Reference Segment Audit

Initial read-only report: `artifacts/ref_audit/report.md`.

- conclusion: `LEAK-FOUND`
- image/garment/hair position-ID grids are disjoint
- ControlNet residual RMS on both reference segments is exactly zero
- decoded reference content has negligible far-outside content contamination
- background image queries allocate 0.2237 attention mass to reference segments
- old reference canvas was RGB 127 while mannequin parsing-background median is
  RGB (228, 227, 226)

The concrete repair is isolated to the two reference cache routes: erode garment
and hair masks by two pixels and use input gray 227. A VAE calibration over gray
values 0/64/96/127/160/192/227/255 confirmed near one-to-one reconstruction;
gray 227 reconstructs to approximately RGB (225, 225, 225). No position offset
change is needed because all three grids are already disjoint.

The audit now infers the intended canvas value from pixels outside each mask
instead of hard-coding the historical value 127. The repaired-cache audit is
written to `artifacts/ref_audit_quality_repair/` before training starts. Its
final conclusion is `NO-LEAK`:

- raw background-to-reference attention mass: 0.23059
- background / foreground reference-attention enrichment: 1.00189
- configured-gray / mannequin-background maximum channel delta: 1
- ControlNet reference-segment residual RMS: 0
- decoded far-outside changed-pixel ratio: 0.0000555

The raw mass is retained in the report but is not a leakage decision by itself:
the two reference segments append 3840 keys, so a substantial aggregate softmax
mass is expected even when background queries do not preferentially read them.
It is considered risky only when background attention is enriched relative to
cloth/face/hair queries, or when the neutral canvas is visibly mismatched.

An offset sweep also ruled out position-ID distance as a useful repair:

| garment/hair offset | background reference mass |
|---:|---:|
| 96 | 0.2313 |
| 128 | 0.2297 |
| 192 | 0.2286 |
| 256 | 0.2292 |

The variation is noise-scale and provides no monotonic reduction. The trained
checkpoint convention therefore remains unchanged at garment `x+64` and hair
`y+64`; changing it at continuation time would introduce an avoidable
train/inference mismatch.

## Part B: High-Frequency Baseline

Comparison report:
`eval/metrics_v2/compare_spatial_hair_incontext_fixedset_vs_a4.md` under the
dataset root.

| metric | spatial step 4400 | A4 | difference |
|---|---:|---:|---:|
| Garment-DINO (higher) | 0.9296 | 0.7433 | +0.1863 |
| Garment-HF-LPIPS (lower) | 0.2201 | 0.5045 | -0.2843 |
| Garment-Gradient-Sim (higher) | 0.4415 | 0.2525 | +0.1890 |
| Print-region HF-LPIPS (lower) | 0.3152 | 0.5309 | -0.2157 |

All paired Wilcoxon p-values above are effectively zero. Spatial conditioning
is substantially better than A4, but the print panel still fails original-item
readability: crest microtext, the Adidas wordmark, Calvin Klein lettering, and
the strokes in "Happy" are visibly blurred or reduced in contrast. This is the
manual high-frequency baseline, not a claim inferred from DINO.

## Part C: Implemented Continuation

- reference inputs: gray 227, two-pixel inward mask erosion; only
  `garment_ref_latents`, `hair_ref_latents`, and `hair_ref_empty` are rebuilt
- repaired references live in the isolated
  `phase1/cache_spatial_quality_repair_768x1024` cache; the historical spatial
  cache is restored with its original gray-127/un-eroded references
- paired MSE: mean-normalized `w_cloth=2`, `w_hair=2`, `w_face=2`; face excludes
  hair to avoid double weighting
- identity: semihard bank-v2 CF pairs, A2 losses unchanged, directed ArcFace
  loss `lambda_dir=0.1`, absolute grounding `lambda_abs=0.05`, decode every
  three optimizer steps in tau [0.35, 0.70]
- evaluation-only AdaFace is not imported by training code
- hair: in-context route only; legacy token route contributes no sequence token;
  DINO loss uses `lambda_hair=0.5`, decode every two optimizer steps, tau
  [0.30, 0.75], minimum projected area 0.5%
- hair logging separates decode-frequency, tau-window, area, invalid-target, and
  parser-failure skips; decode-attempt skip target is below 15%
- watcher: immutable 16-sample protocol, medians, absolute and relative guards,
  warning on one failed checkpoint and stop only after repeated failures
- watcher trends include Garment-HF-LPIPS, Garment-Gradient-Sim, Hair-DINO/LAB,
  directed `sim_gap`, face detection, and identity swap response

Resolved continuation config: `configs/spatial_quality_repair_resume.yaml`.

## Verification And Execution

The repaired reference cache was rebuilt in four shards with zero failures, the
eight-sample audit passed, and project tests pass: `69 passed` with
`pytest -q tests`. Running pytest without the `tests/` boundary also collects
vendored baseline repositories under `comparisons/`; their missing independent
dependencies are not project failures.

The repaired-input step-4400 fixed-set baseline has no generation failures:
Garment-DINO median `0.755753`, Garment-HF-LPIPS median `0.385374`,
Hair-DINO median `0.459795`, head5 median `0.023175`, body median
`0.012192`, face detection `100%`, and maximum paired/swap face cosine
`0.317787`. The fixed val set and 400-image system suite have different metric
distributions, so watcher training gates use fixed-baseline minus tolerance
(`0.755753 - 0.02`) while the full-eval Garment-DINO target remains `0.92`.
Mixing the full-eval absolute value into the fixed-set watcher would reject the
unchanged baseline and recreate the measurement-protocol false stop.

The strict 44 GiB admission gate was replayed with the formal three-rank DDP,
`grad_accum=6` topology. Frozen training-side DINO in BF16 was insufficient by
itself: decode scale `0.50` peaked at `44.014976 GiB`, so that launch was stopped
at step 4401 and its output directory was quarantined without a checkpoint.
Reducing only the sparse differentiable-decode latent scale to `0.48` produced
`43.826136 GiB` on the same seed and sampler state. ArcFace and DINO still see
their fixed 112x112 and 518x518 inputs. A separate 20-step worst-case smoke
completed at `43.654990 GiB`, with all three transformer forwards, both sparse
decode objectives, face-weighted MSE, and hair skip accounting active. The
formal run is therefore admitted with approximately 178 MiB of measured
headroom; its first optimizer step remains a mandatory live VRAM check.

Detailed measurements:
`docs/results/vram_report_spatial_quality_repair_r16_768x1024.md`.

Execution order:

1. rebuild only reference cache keys on four GPUs
2. rerun the eight-sample reference audit
3. backfill step 4400 on the frozen watcher set under the repaired input protocol
4. run the 20-step single-GPU directed smoke and write peak VRAM/throughput report
5. continue step 4400 to 6400 with GPU0-2 and watcher on GPU3
6. run frozen-subset metrics-v2 and qualitative print/face/hair panels
7. decide rank 32 only from the preregistered Part D conditions

Formal launcher: `scripts/run_spatial_quality_repair.sh`.

## Part D Status

Not decided before Part C. Rank 32 is triggered only if post-continuation print
fidelity has no meaningful improvement, held-out `sim_target` remains below
0.40, or the two objectives each improve only partially.
