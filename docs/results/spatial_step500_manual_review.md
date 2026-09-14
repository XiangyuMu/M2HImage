# Spatial Warmup Step-500 Review

Date: 2026-08-17

## Human Gate

The mandatory review did not pass:

- Garment: only a coarse silhouette is preserved; print, cut, and fine placement are not reliably the same item.
- Hair: only a coarse outline is preserved; identity-specific hair is not reliably reproduced.
- Head orientation: follows the mannequin ground truth and passes the visual check.

The run remains paused at \`step-000500\`. No approval file was written and no checkpoint was modified.

## Automatic Checks

- Face detection: \`10/10 = 100%\`.
- Swap ArcFace cosine: \`0.4930\` and \`0.2356\` (both below the \`0.85\` non-response threshold).
- Condition gates moved from \`0.1\`:
  - appearance: max deviation \`0.00594\`;
  - hair: max deviation \`0.00811\`;
  - head pose: max deviation \`0.01242\`.

## Read-Only Response Probe

Probe artifact: \`artifacts/spatial_step500_response_region.md\` and its JSON companion.

At the same noise and timestep, replacing the garment reference produced a mean velocity response of
\`0.2025\` relative to the baseline velocity (two validation samples, \`tau=0.5\`), about \`5.81x\` the
identity-swap response. The response was approximately uniform across cloth, face, and body/background
regions, so the route is connected but is not yet spatially localized.

Hair replacement or disabling produced only \`0.0133-0.0136\` relative response, about \`0.38-0.39x\`
the identity-swap response. This is a weak route and is consistent with the Phase-1 hair loss being
disabled; it must not be reported as a passed hair-condition gate.

## Decision

Do not approve step 500. Do not resume the 4400-step run until a follow-up decision is made about the
hair supervision/conditioning strength and the garment localization objective. The existing checkpoint
is preserved for controlled comparison.
