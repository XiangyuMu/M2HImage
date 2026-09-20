# Role-Flow A/B/C and C2 experiment record

This branch records the implementation used for the M2H mannequin-to-human role-flow experiments. The frozen A/B/C results are the confirmatory baseline; `C2_tempered_async_flow` is a single post-hoc exploratory follow-up selected after inspecting those results. C2 must not be merged into the frozen A/B/C table or used to select a further C3 on the same test set.

## Structures

- **A — timestep-routed adapters:** the standard FLUX flow with timestep-routed condition adapters. It changes where identity, pose, garment, and spatial conditions enter the denoiser while retaining the global flow path.
- **B — garment fidelity weighting:** A plus paired-region flow-loss weighting (`w_cloth=2.5`, `w_hair=1.0`, `w_face=1.0`). This emphasizes mannequin-derived garment supervision without changing the sampling path.
- **C — asynchronous flow path:** B plus endpoint-preserving region-wise flow targets. The frozen schedule is background `-0.5`, garment `-0.5`, boundary `0.0`, identity `0.3`; the corrected velocity target preserves the endpoints while allowing regions to follow different paths.
- **C2 — tempered asynchronous flow:** C's schedule is multiplied by one half: background `-0.25`, garment `-0.25`, boundary `0.0`, identity `0.15`. The model, paired weights, split, cache, seed, and 4,400-step budget remain unchanged. `experiment_method.name` stays `C` because that name enables the asynchronous implementation in the training code.

## Frozen protocol

The formal evaluator uses the exact 580-row generation manifest (180 validation rows for preflight and 400 final-test rows for metric aggregation), mannequin-only FASHN garment masks, no generated-mask fallback, the fixed AdaFace FAR@1e-3 threshold, and a 1,971-image final-test human FID reference. Generated images with no detectable face receive an explicit conservative identity outcome (`id_cosine=-1`, TAR=0) and are counted in provenance; no embedding is fabricated.

The source and evaluator hashes are recorded in the run provenance. The current protocol-v2 evaluator also rejects stale files in generated directories, creates the pose output directory before inference, and passes the torch-fidelity feature layer as the required string `"2048"`.

## Frozen A/B/C means

These values come from `role_flow/summary_v2/role_flow_abc_summary.json`, not from the older smoke evaluator:

| Metric | A | B | C | Direction |
|---|---:|---:|---:|---|
| ID cosine | 0.259325 | 0.252816 | 0.247061 | higher |
| TAR@1e-3 | 0.001724 | 0.001724 | 0.000000 | higher |
| Garment DINO | 0.804022 | 0.864590 | 0.849368 | higher |
| Garment IoU | 0.867611 | 0.895645 | 0.891639 | higher |
| Pose PCK | 0.973744 | 0.979620 | 0.973640 | higher |
| BG SSIM | 0.975871 | 0.976335 | 0.977589 | higher |
| BG LPIPS | 0.109851 | 0.105174 | 0.099980 | lower |
| FID | 55.3343 | 56.0256 | 58.1410 | lower |

The evidence is a trade-off: A leads identity and FID, B leads garment and pose, and C leads background preservation. The data do not support a claim that C dominates B and A. C2 is intended to test whether reducing the asynchronous path amplitude recovers the B-side garment/identity trade-off while retaining part of C's background gain.

## Reproducibility

The C2 config is `configs/role_selective/C2_tempered_async_flow.yaml`. Its exact hash and the pre-registered exploratory gates are recorded in the remote experiment registration artifact, alongside split/cache, TAR calibration, FID reference, checkpoint, and evaluator hashes. Formal outputs live outside the source repository under the versioned experiment root so generated images and checkpoints are not committed to Git.
