# G2 source-ownership annotation protocol v1

Status: frozen annotation draft; empirical G2 remains unpassed until three independent annotators and an independent verifier complete the gate.

Planning authority is limited to `.omx/plans/prd-cvpr-source-attributed-m2h.md` and `.omx/plans/test-spec-cvpr-source-attributed-m2h.md`.

## Purpose

Label region × feature-family units so that identity-exclusive response, mannequin/context-exclusive response, legal interaction/occlusion, and uncertainty are not conflated. This protocol is evaluation-only and cannot be loaded by training or inference routing.

## Labels

- `S_I`: identity-exclusive. Inner-face morphology and reliable local hair appearance/geometry follow the identity source.
- `S_M`: mannequin/context-exclusive. Global head location/scale/pose, body morphology/pose, visible garment design, gross garment drape, and scene follow the mannequin/context source.
- `S_X`: interaction/occlusion. Hair–face contact, hair–garment contact, contact folds, and visibility/order effects can respond to both inputs.
- `S_U`: uncertain/unclaimed. Excluded from exclusive leakage and interaction-coherence claims.

Exactly one support label is assigned to every annotated unit. `S_X` and `S_U` never enter exclusive leakage calculations.

## Independent annotation

Three annotators label the same frozen image/unit inventory independently. Do not inspect another annotator's sheet. For each unit provide:

1. `support_label` in `{S_I,S_M,S_X,S_U}`;
2. `occlusion_order` for occlusion units (`none`, `hair_over_face`, `face_over_hair`, `hair_over_garment`, `garment_over_hair`, or `ambiguous`);
3. whether the owner is observable;
4. a positive unit weight used only for the adjudicated `S_U` coverage calculation;
5. a polygon in canonical 768×1024 coordinates as JSON `[[x1,y1],...]`;
6. artifacts and notes.

After all three sheets are locked, a blinded adjudicator resolves ties and records the reason without altering raw labels.

## Hair decision gate

Keep the main-hair policy only if all conditions pass:

- nominal Krippendorff alpha ≥ 0.80;
- median pairwise boundary IoU ≥ 0.75;
- occlusion Fleiss kappa ≥ 0.70;
- adjudicated `S_U` share ≤ 10% of head/hair unit weight in at least 90% of images.

If the first pass fails, revise this guideline once and repeat the annotations. If the revised pass fails, freeze the face-only policy: inner face remains `S_I`, unsupported hair becomes `S_U`, and the face/head boundary becomes `S_X`. The main-hair and face-only policies cannot be mixed in a primary test.

## Separation and leakage safety

Conditioning supports may use only `M`, `I`, and frozen automatic preprocessors. Evaluation supports must use an independently frozen pipeline plus blinded annotations. Output masks, generated images, photographic targets, manual evaluation masks, and evaluator annotations are forbidden routing inputs.

The existing FASHN SegFormer-B4 pipeline is the first evaluation parser. The existing SAM-derived `cloth_safe` masks are not an independent second parser. G4 cannot pass until a genuinely independent second parser is frozen and preserves the main effect directions.

## Current stop condition

Passing code tests only validates the protocol machinery. G2 remains open until completed raw sheets, adjudication, hashes, the threshold report, and an independent verifier PASS exist. No 2×2 generation and no new-method training may begin before that gate.
