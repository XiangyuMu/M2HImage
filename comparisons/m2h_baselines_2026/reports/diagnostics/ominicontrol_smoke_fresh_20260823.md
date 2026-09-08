# OminiControl fresh M2H smoke (2026-08-23)

## Outcome

Status: runnable smoke gate passed. OminiControl was initialized from the
configured FLUX.1-dev pretrained model and freshly trained with three M2H
adapters. It produced all four fixed counterfactual outputs, and Metrics-v2
completed with full coverage and no warnings. Visual quality remains recorded
for comparison and is not used as a blocker.

## Data and task provenance

- Paired training manifest: 32 records.
- Counterfactual inference manifest: 4 records.
- Spatial conditions: `agnostic/mannequin(mid)` and `pose(mid)`.
- Identity condition: `identity_card(jid)`.
- Target: `human(mid)` for paired training supervision only.
- Frozen protocol SHA256:
  `7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985`.

## Training

- Schedule: 100 optimizer steps (800 batch steps with accumulation 8), seed 42,
  512x512, two RTX 3090 GPUs, FSDP.
- Trainable parameters: 24,552,192 LoRA parameters.
- Wall time to the batch-step-800 save: about 1 hour 53 minutes.
- Log:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/ominicontrol/smoke/logs/20260823-104258.log`.
- Checkpoint:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/ominicontrol/smoke/20260823-104323/ckpt/800`.
- Checkpoint verification: pass (`task=mannequin_to_human`, global step 100,
  batch step 800, current protocol hash, explicit condition contract, all
  three adapters present).
- `m2h_checkpoint.json` SHA256:
  `520e8575d5d14415164f898dd623533570a60ecfbe6e1e9929fbcb4303cb566f`.

Adapter SHA256 values:

- `mannequin.safetensors`:
  `b06c33a364102d0123e20d5531862aa7b7821ef3913346d0b971942d34cb7014`
- `identity.safetensors`:
  `a412ac4087ed2f8ab0e11b53d5eba66b6f34644712efd2183ed9ec65ed9edb03`
- `pose.safetensors`:
  `ed5a0a5af8fcec72a023669a99209d85558e52307126722daf2cba08fee242f7`

## Counterfactual inference

- Canonical records: 4/4.
- Inference log:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/ominicontrol/infer_smoke/logs/20260823-123545.log`.
- Inference manifest SHA256:
  `1781d7c2f569fb532dddc3d54e5c2a6704c1ede4d94e41ea63d7d73cd34d0d5d`.
- Canonical root:
  `/data/muxiangyu/programs/M2H_Baselines_2026/outputs_task_retrain_20260822/ominicontrol/smoke/canonical`.

Raw image SHA256 values:

- `0000_01968_35295_s0.png`:
  `dfead839f6b47d7accbcf8b8cf72d566876e78f6517f96dbbb26ed69035a6e91`
- `0000_01968_35295_s1.png`:
  `315d2894ee97c35b0453cd0d5a3cd48d430d6feedbe8a3cd1480cd3a53e216bb`
- `0001_01968_18372_s0.png`:
  `57d670c833dcd8d2d64d3827935bc443d87dfb0f7b4cb3d14abb128d0ef73edc`
- `0001_01968_18372_s1.png`:
  `5330fae91a087f5ccee0ff6bff75b4eecfe33d4084580de1869ae6f0934a408d`

## Metrics-v2

Verification: `status=ok`, `quality_status=ok`, expected count 4,
parsing/garment/hair/pose/identity all 4/4, panels 2, errors empty, and no
quality warnings.

- Garment-DINO-to-mannequin: 0.8418.
- Garment-HF-LPIPS: 0.4148.
- Hair-DINO-to-reference: 0.3309.
- DWPose head-5 distance: 0.0224.
- Face detection rate: 4/4.
- Held-out deltaID: -0.0411 (detected-face rows only).
- Report:
  `/data/muxiangyu/programs/M2H_Baselines_2026/reports_task_retrain_20260822/generated/ominicontrol/smoke/ominicontrol_smoke/report.md`.
