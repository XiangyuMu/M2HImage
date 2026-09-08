# IDM-VTON fresh M2H smoke (2026-08-22)

## Outcome

Status: runnable smoke gate passed. The model was initialized from the official
IDM-VTON weights and freshly trained for the M2H task; no historical comparison
checkpoint was reused. Visual quality and metric values are recorded but do not
block subsequent methods.

## Data and task provenance

- Paired preflight: 32 records, `source` shape `(3, 512, 512)`, identity CLIP
  input shape `(1, 3, 224, 224)`.
- Counterfactual preflight: 4 records with the same condition contract.
- Source: `mannequin(mid)`.
- Identity: `identity_card(jid)`.
- Target: `human(mid)` (paired training supervision only).
- Frozen protocol SHA256:
  `7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985`.

## Training

- Schedule: 100 optimizer steps, seed 42, 512x512, two RTX 3090 GPUs, FSDP,
  effective batch size 16.
- Wall time reported by the trainer: 46 minutes 30 seconds.
- Log:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/idm_vton/smoke/logs/20260822-184811.log`
- Checkpoint:
  `/data/muxiangyu/programs/M2H_Baselines_2026/runs_task_retrain_20260822/idm_vton/smoke/checkpoint-100`
- UNet SHA256:
  `0de9be3092f9e6781b5e8f60c31db9b91e11ad26e03156ec2f7d443537dc2bbd`.
- `m2h_checkpoint.json` SHA256:
  `3dcb7a295e7f667ca133d32cd121a6a87eb24745afdebed92eab3628db3de622`.
- Checkpoint verification: pass (`task=mannequin_to_human`, global step 100,
  current protocol hash, complete UNet, explicit condition contract).

## Counterfactual inference

- Canonical records: 4/4.
- Inference manifest SHA256:
  `ece40721d5a573b447aa80d2e286a532946c6ffac285f50bf1f6193972787efd`.
- Canonical root:
  `/data/muxiangyu/programs/M2H_Baselines_2026/outputs_task_retrain_20260822/idm_vton/smoke/canonical`.
- Raw image SHA256 values:
  - `0000_01968_35295_s0.png`: `f9ff47349fc3fe7444d9bd926f5e3901243001d30d34fb847ca6b71e5666c68d`
  - `0000_01968_35295_s1.png`: `4d4194c093ca42e7d57252039a7de7637f1314f01a4e49f257b7185bc981d5eb`
  - `0001_01968_18372_s0.png`: `ae5ca0b0e712b0556e86b52bc121013d6a594851a705fea1c78181857a8b09cb`
  - `0001_01968_18372_s1.png`: `562efdd00f7e9fe89368716f110736f75f8f26d6ab05cf406c45b548ba16fa52`

## Metrics-v2

Verification: `status=ok`, expected count 4, all parsing/garment/hair/pose/
identity checks have 4 valid records, panels 2, errors empty.

- Garment-DINO-to-mannequin: 0.9618.
- Garment-HF-LPIPS: 0.1348.
- Hair-DINO-to-reference: 0.4465.
- DWPose head-5 distance: 0.0069.
- Face detection rate: 4/4.
- Report:
  `/data/muxiangyu/programs/M2H_Baselines_2026/reports_task_retrain_20260822/generated/idm_vton/smoke/idm_vton_smoke/report.md`.
