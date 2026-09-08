# ITA-MDT smoke stage 4 diagnosis (2026-08-22)

## Outcome

Study status: runnable smoke gate passed. The visual assessment below remains a
diagnostic observation, but it no longer blocks subsequent comparison methods.
Metrics-v2 was subsequently run to completion and IDM-VTON was started.

This supersedes the earlier study-level decision boundary at the end of this
report. The acceptance criterion is now checkpoint creation, counterfactual
output completion, canonical collection, and an error-free Metrics-v2 run;
generation quality is reported but is not a smoke gate.

The stage-4 model improves some facial landmark placement relative to stage 3,
but none of the four fixed counterfactual outputs passes the joint Face/Hair
visual gate. All four retain a generated shoulder patch and a hard upper-arm
boundary. Continuing the same fine-tuning schedule is not justified by the
available evidence.

## Frozen protocol

- Protocol SHA256:
  7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985
- Counterfactual manifest SHA256:
  da54b106e236332580c97da1b8c9a1644e6fc94cab8606b908109dea29b0a455
- Stage-4 inference manifest SHA256:
  8c28c7adf5392c846178ecad473e41f09c1a8e01311ce110f71caef7c23515d4

## Stage-4 change

The former loss normalized the first channel of a VAE-encoded keep-mask latent
and treated it as a spatial mask. Stage 4 instead:

- carries the augmented disk M_replace mask as an explicit one-channel binary
  replace_mask;
- keeps agn_mask as the VAE-encoded keep condition used by the model;
- downsamples replace_mask with nearest-neighbour sampling;
- normalizes masked MSE by selected pixels and channels.

Regression results:

- ITA replace-mask tests: 3/3 pass;
- comparison project tests: 4/4 pass;
- paired and counterfactual preflight: pass;
- paired replace fraction: 0.14466094970703125;
- counterfactual replace fraction: 0.07547760009765625.

## Training

- Initial model:
  /data/muxiangyu/programs/M2H_Baselines_2026/runs_dense_pose/ita_mdt/smoke/model001000.pt
- Output:
  /data/muxiangyu/programs/M2H_Baselines_2026/runs_replace_loss/ita_mdt/smoke
- Schedule: 1000 steps, global batch 2, two RTX 3090 GPUs, fresh optimizer,
  fresh EMA, fresh step.
- Log:
  /data/muxiangyu/programs/M2H_Baselines_2026/runs_replace_loss/ita_mdt/smoke/logs/20260822-121802.log

Checkpoint SHA256:

- model001000.pt:
  9aa19176241c76c13888ce7e88003dfb09ff3bd47ef538fe8c0c6e8e89f2b1ca
- ema_0.9999_001000.pt:
  beb26b4b68b3a5d162c7998c53f1bb953983d4437fe5ece763c6f539caf4b444
- opt001000.pt:
  41164cfd1d590ab58276786ea522d7924cc093ba7281575d7b4c1791e9fe65e4
- opt001000.rank1.pt:
  9aa3e581edb1b33494d09711ecd8c04de201bfd7c055a31a730f1e42b5ae236c

## Fixed counterfactual inference

Inference explicitly used the non-EMA model001000.pt, 30 DDIM steps, and the
same four sample names and seeds as stage 3.

Output SHA256:

- 0000_01968_35295_s0.png:
  90467a4149cf7bc19bcd34f021f3d43efee62ad076c36925260a952928c867fa
- 0000_01968_35295_s1.png:
  f686383f1a6ba179262c7f4c81cf41ba207015654f530766c3161c55bfaaf672
- 0001_01968_18372_s0.png:
  1fa537fe2b743e9db0bfbabe88b541a52bfea63978d8f6d026995de599de35ff
- 0001_01968_18372_s1.png:
  8b526504414a0e5de1a1e8b8ff7552a0d8ba61a0b8985509a0e734a2a78e66df

Stage-3 outputs were preserved before stage-4 inference at:

/data/muxiangyu/programs/M2H_Baselines_2026/outputs/ita_mdt/smoke_stage3_dense_pose

## Root-cause probes

Paired probe:

/data/muxiangyu/programs/M2H_Baselines_2026/diagnostics/ita_mdt/paired_probe_stage4

The first four training identities reconstruct successfully. This rules out a
broken checkpoint, pose loader, or general failure to synthesize a person.

Full-human identity probe:

/data/muxiangyu/programs/M2H_Baselines_2026/diagnostics/ita_mdt/counterfactual_full_human_probe_stage4

Replacing the fragmented identity_person cloth condition with the allowed full
human(jid) reference does not repair counterfactual faces. It also introduces
reference-garment leakage at the neck and shoulder.

## Conclusion

ITA-MDT's DINO cloth pathway can copy a same-pose paired identity after
fine-tuning, but it does not provide a reliable identity-geometry mechanism for
cross-person, cross-pose M2H replacement. A softer repaint edge can hide the
white seam but cannot repair facial identity and therefore cannot make this
smoke result pass.

Under the original visual gate this would have required a study-level decision.
Under the current runnable-comparison criterion, ITA-MDT is retained as a
completed baseline with weak counterfactual quality, and work continues with
IDM-VTON without changing ITA-MDT into a new identity-adapter method.
