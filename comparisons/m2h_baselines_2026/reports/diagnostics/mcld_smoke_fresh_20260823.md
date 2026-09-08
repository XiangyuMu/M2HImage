# MCLD fresh M2H smoke (2026-08-23)

## Outcome

Status: runnable smoke gate passed. MCLD was initialized from the configured
official pretrained weights and freshly trained for the mannequin-to-human task.
It produced all four fixed counterfactual outputs and Metrics-v2 completed with
full artifact coverage. Generated-image quality is recorded but does not block
OminiControl.

## Data and task provenance

- Paired training manifest: 32 records.
- Counterfactual inference manifest: 4 records.
- Spatial condition: texture, pose, and face region from `mannequin(mid)`.
- Identity condition: antelopev2 face embedding from `human(jid)`.
- Target: `human(mid)` for paired training supervision only.
- Frozen protocol SHA256:
  `7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985`.

## Training

- Schedule: 100 optimizer steps, seed 42, 512x512, two RTX 3090 GPUs, FSDP,
  effective batch size 16.
- Trainer elapsed time: about 22 minutes 18 seconds.
- Log:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/mcld/smoke/logs/20260822-222452.log`.
- Checkpoint:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/mcld/smoke/mcld_m2h_smoke`.
- Checkpoint verification: pass (`task=mannequin_to_human`, global step 100,
  current protocol hash, explicit condition contract, all four components
  present).
- `m2h_checkpoint.json` SHA256:
  `6ae0e0e44b470da7121dd67a2b837e20a40387cd74462a851f5dfc1af1e4b573`.

Component SHA256 values:

- `reference_unet-100.pth`:
  `25cd6ba1157e578db43b34376b744cd7c326751b55708c4fc263317798aaf471`
- `denoising_unet-100.pth`:
  `0c79825f0fc75376ca7230fa4f789613e68af218c058740a70107064beae3c6f`
- `pose_guider-100.pth`:
  `934080839bd7f180ff44acba7d719eb6652c54270ee17f4bc76eabd3fcd06480`
- `image_proj_model-100.pth`:
  `df9b3640ea26144b7fb6d112b8b0a08d30824664c94b8f9dc1dbc9f083b93da2`

## Counterfactual inference

- Canonical records: 4/4.
- Inference manifest SHA256:
  `4b6e4556b74426ca85ede3abc61520ff2c9bb594151362120aa716d44b8fb2c0`.
- Canonical root:
  `/data/muxiangyu/programs/M2H_Baselines_2026/outputs_task_retrain_20260822/mcld/smoke/canonical`.
- Inference log:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/mcld/infer_smoke/logs/20260823-102656.log`.

Raw 512x512 image SHA256 values:

- `0000_01968_35295_s0.png`:
  `6b61fbf26b2a63efe9d95273bafba118e3eef8038254ed311338d67fe2e68036`
- `0000_01968_35295_s1.png`:
  `aa28ac6dc02cb2183c44e83694af50b475672501ac9ecf60ee324e48f168e05e`
- `0001_01968_18372_s0.png`:
  `b28f5b3fce86d43f3a58ab42387f68b3e81bb1493c14f4432ef72b27c57ff084`
- `0001_01968_18372_s1.png`:
  `ede3a801c41411f71df41da3684fa28231ac941053a3232b62025c4653ee0d07`

## Metrics-v2

Execution verification: `status=ok`, expected count 4, parsing/garment/hair/
pose CSV coverage 4/4, panels 2, structural errors empty.

Quality status: warning. InsightFace detected faces for 2/4 images; the two
`jid=35295` rows remain explicitly marked non-ok. This is a generated-image
quality result, so it is retained rather than repaired, filtered, or treated as
an infrastructure failure.

- Garment-DINO-to-mannequin: 0.2922.
- Garment-HF-LPIPS: 0.5133.
- Hair-DINO-to-reference: 0.3525.
- DWPose head-5 distance: 0.0500.
- Face detection rate: 2/4.
- Held-out deltaID on detected faces: 0.0022.
- Report:
  `/data/muxiangyu/programs/M2H_Baselines_2026/reports_task_retrain_20260822/generated/mcld/smoke/mcld_smoke/report.md`.
