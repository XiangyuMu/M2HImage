# P0 input-condition audit — 2026-09-07

Scope: read-only code tracing in `/data/muxiangyu/pythonPrograms/M2HImage`
on `10.249.45.227`. No G2 labels or human review required.

## Confirmed data paths

| Feature | Existing source | Decision for new protocol |
|---|---|---|
| `pose_latents` | `build_cache.py:130–138`: M-side DWPose with/without head | M-derived, permitted; regenerate under new provenance or audit original construction |
| `pose_synth_latents` | `build_cache.py:140–151` → `synth_head_keypoints.py:_theta`: H_mid head angles | Not input-only; regenerate from M or use declared missing-head policy |
| `head_pose` | `build_cache.py:297–298` → `conditions.py:168–183`: H_mid 6DRepNet JSON | Not input-only; remove access to H_mid, use M-derived estimate or null token |
| `garment_grid` | `build_cache.py:192–201` → `conditions.py:garment_crop`: M RGB with H_mid SAM mask | Recompute using M-only parsing/mask |
| `garment_ref_latents` | `build_cache.py:202…` → `spatial_conditions.py:30–55`: M RGB with H_mid SAM mask | Recompute using M-only mask |
| `target_latents` | `build_cache.py:125–129`: H_mid | Permitted paired training target only; absent from inference payload |
| PuLID/appearance/hair_j | `eval_b2.py:185–214`: identity j cache | Permitted iff provenance traces only to I_j; do not confuse H_j reference with H_mid target |
| text | `eval_b2.py:189,196–197`: prompt cache | Permitted if frozen prompt contains no target metadata |

`eval_b2.make_cf_batch` does **not** read `target_latents` itself. Its defect
for the new information boundary is the provenance of head/garment cached
conditions, not a claim that it copies the target RGB at inference.

`build_cache.py:118` resolves `images/human/{id}` even for M-only requested
keys. Thus a strict M-only cache builder must not blindly call that routine:
it needs role-separated path resolution and no implicit target fallback.

`paired_eval_common.py` is primarily metric/report comparison code, not the
generation entry point; implementation should start at
`eval_b2.make_cf_batch` plus the new condition builder. Existing signatures
should remain backwards compatible and legacy caches/results unchanged.

## Existing resources and remaining prerequisites

- Existing source parser implementation: `metrics_v2/parsing.py:FashnParser`,
  local FASHN SegFormer weights; it accepts RGB arrays and needs no H_mid.
- Dataset contains `human_parsing/fashn/masks/mannequin`; provenance needs
  verification before reuse. B-D0 deliberately recomputes from input RGB.
- Existing `identity_bank_v2.npz` has 36,034 IDs ×512 embeddings, Glint360K
  ArcFace, matching train count only. It is **not** an all-split identity
  grouping audit. Val/test and uncertain identity relations remain uncovered.
- Scoped inventory of `modelLibrary/insightface/models` and project
  `models/local` found `antelopev2/glintr100.onnx` plus detector/landmark/age
  networks, no independent dev recognizer in those searched locations.
  This is not evidence that no such weights exist elsewhere. Do not use
  final AdaFace to fill the gap silently.

## Required implementation and fresh verification

1. Generate M-only garment mask, garment CLIP/VAE, M pose and declared head
   missingness into a new cache namespace; no reuse of H_mid-derived keys.
2. Separate M and I cache roles; provenance includes model/input hashes.
3. Build 32-example inference manifest with M/I only and a file-read guard
   that denies H_mid and old unverified cache paths. Audit all subprocesses;
   a Python-only hook is not by itself proof of subprocess isolation.
4. Confirm all 32 outputs and successful attempted-denial tests, with no
   denied reads needed by normal inference. Do not infer a pass merely from
   output quality being similar to the historical model.

Current verdict: **input-only inference gate NOT PASSED**; this is a
recoverable implementation prerequisite, not a reason to restart G2 or
ask for human annotations. B-D0 source-only diagnostics can proceed safely.
