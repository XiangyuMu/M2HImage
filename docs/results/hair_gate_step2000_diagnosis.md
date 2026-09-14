# Hair Gate Step-2000 Inspection

**Conclusion: `INCONCLUSIVE`.** wiring and gradient are live; one dry-run cannot explain the flat gate

- Checkpoint: `/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_region_resume_r16_4400_768x1024/checkpoints/step-002000` (step 2000)
- Real batch sample: `00001`, tau=0.5
- Hair gate: value=0.100031823, requires_grad=True, group={'index': 2, 'name': 'condition_gates_fp32', 'lr': 0.0005625000000000001, 'weight_decay': 0.0}, grad={'is_none': False, 'value': -0.00037216360215097666, 'abs': 0.00037216360215097666, 'l2': 0.00037216360215097666, 'finite': True}
- Appearance control: value=0.118405528, grad={'is_none': False, 'value': 0.00033924123272299767, 'abs': 0.00033924123272299767, 'l2': 0.00033924123272299767, 'finite': True}
- Hair/appearance max-gradient ratio: `1.097047075214649`
- Projected hair tokens: `{'shape': [1, 64, 4096], 'l2': 316.2608642578125, 'rms': 0.61769700050354, 'nonzero_fraction': 0.327911376953125, 'finite': True}`
- Post-gate hair tokens: `{'shape': [1, 64, 4096], 'l2': 29.345584869384766, 'rms': 0.05731559544801712, 'nonzero_fraction': 0.328125, 'finite': True}`
- Transformer encoder condition slice: `[516, 580]`; stats=`{'shape': [1, 64, 4096], 'l2': 29.345584869384766, 'rms': 0.05731559544801712, 'nonzero_fraction': 0.328125, 'finite': True}`
- Gate order: `post_layernorm`
- Peak allocated VRAM: 34.389 GiB

The checkpoint and optimizer were loaded read-only. The dry-run called backward but did not call optimizer.step().

```json
{
  "conclusion": "INCONCLUSIVE",
  "reasons": [
    "wiring and gradient are live; one dry-run cannot explain the flat gate"
  ],
  "checkpoint": "/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_spatial_hair_region_resume_r16_4400_768x1024/checkpoints/step-002000",
  "checkpoint_step": 2000,
  "config": "/data/muxiangyu/pythonPrograms/M2HImage/configs/spatial_warmup_resume_hair.yaml",
  "sample_id": "00001",
  "tau": 0.5,
  "loss": 0.3759996294975281,
  "hair_gate": {
    "value": 0.10003182291984558,
    "requires_grad": true,
    "optimizer_group": {
      "index": 2,
      "name": "condition_gates_fp32",
      "lr": 0.0005625000000000001,
      "weight_decay": 0.0
    },
    "grad": {
      "is_none": false,
      "value": -0.00037216360215097666,
      "abs": 0.00037216360215097666,
      "l2": 0.00037216360215097666,
      "finite": true
    }
  },
  "appearance_gate_control": {
    "value": 0.1184055283665657,
    "requires_grad": true,
    "optimizer_group": {
      "index": 2,
      "name": "condition_gates_fp32",
      "lr": 0.0005625000000000001,
      "weight_decay": 0.0
    },
    "grad": {
      "is_none": false,
      "value": 0.00033924123272299767,
      "abs": 0.00033924123272299767,
      "l2": 0.00033924123272299767,
      "finite": true
    }
  },
  "gradient_ratio_hair_to_appearance": 1.097047075214649,
  "hair_route": {
    "gate_order": "post_layernorm",
    "raw_projected_tokens": {
      "shape": [
        1,
        64,
        4096
      ],
      "l2": 316.2608642578125,
      "rms": 0.61769700050354,
      "nonzero_fraction": 0.327911376953125,
      "finite": true
    },
    "post_gate_tokens": {
      "shape": [
        1,
        64,
        4096
      ],
      "l2": 29.345584869384766,
      "rms": 0.05731559544801712,
      "nonzero_fraction": 0.328125,
      "finite": true
    },
    "valid_token_count": 21.0,
    "adapter_slice": [
      4,
      68
    ],
    "encoder_condition_slice": [
      516,
      580
    ],
    "encoder_condition_slice_stats": {
      "shape": [
        1,
        64,
        4096
      ],
      "l2": 29.345584869384766,
      "rms": 0.05731559544801712,
      "nonzero_fraction": 0.328125,
      "finite": true
    },
    "note": "Hair tokens occupy encoder_hidden_states, not FLUX image hidden_states; the reported condition slice is the actual transformer input route."
  },
  "metrics": {
    "head_pose_null_ratio": 0.0,
    "head_control_synthetic_ratio": 1.0,
    "garment_reference_tokens": 3072.0,
    "tau_mean": 0.5,
    "z1_mean": 0.003939468413591385,
    "controlnet_forward_count": 1.0,
    "appearance_gate": 0.1184055283665657,
    "head_pose_gate": 0.10241937637329102,
    "hair_gate": 0.10003182291984558,
    "loss_total": 0.3759996294975281,
    "loss_pair": 0.3759996294975281,
    "transformer_forward_count": 1.0,
    "loss_pair_unweighted": 0.33258485794067383,
    "mse_cloth_safe": 0.542062520980835,
    "mse_hair": 0.77339106798172,
    "mse_face": 0.6146059632301331,
    "mse_other": 0.25170260667800903,
    "pair_weight_mean": 1.0,
    "pair_weight_max": 1.6072379350662231
  },
  "load_notes": {
    "controlnet": {
      "path": "/data/muxiangyu/pythonPrograms/M2HImage/models/hf/InstantX/FLUX.1-dev-Controlnet-Union",
      "config_hash": "05cd689a748d650e",
      "weight_hash": "2cd23f9da9f2f24d",
      "config": {
        "_class_name": "FluxControlNetModel",
        "_diffusers_version": "0.30.0.dev0",
        "_name_or_path": "/mnt/wangqixun/",
        "attention_head_dim": 128,
        "axes_dims_rope": [
          16,
          56,
          56
        ],
        "guidance_embeds": true,
        "in_channels": 64,
        "joint_attention_dim": 4096,
        "num_attention_heads": 24,
        "num_layers": 5,
        "num_mode": 10,
        "num_single_layers": 10,
        "patch_size": 1,
        "pooled_projection_dim": 768
      }
    },
    "lora": "LoRA rank=16, trainable=26,148,864",
    "adapter": {
      "type": "condition_tokens_no_identity_pulid",
      "version": "phase1-spatial-hair-2026-08-17",
      "identity_route": "disabled; handled by pretrained PuLID-FLUX only",
      "gate_order": "post_layernorm",
      "gate_dtype": "torch.float32",
      "legacy_garment_tokens": false,
      "hair_tokens": true,
      "gates": {
        "appearance_gate": 0.10000000149011612,
        "head_pose_gate": 0.10000000149011612,
        "hair_gate": 0.10000000149011612
      }
    },
    "pulid": {
      "type": "pulid_flux_v0.9.1",
      "repo": "/data/muxiangyu/modelLibrary/PuLID",
      "weight_path": "/data/muxiangyu/modelLibrary/PuLID/models/pulid_flux_v0.9.1.safetensors",
      "weight_hash": "92c41c3af322b02e",
      "antelopev2_dir": "/data/muxiangyu/modelLibrary/PuLID/models/antelopev2",
      "hf_home": "/data/muxiangyu/modelLibrary",
      "double_interval": 2,
      "single_interval": 4,
      "id_weight": 1.0,
      "trainable": 0
    },
    "pulid_ca_self_check_l2": 27.23750114440918,
    "pulid_transformer_self_check_l2": 25.26360511779785,
    "pulid_context_switch_self_check_l2": 25.599828720092773,
    "vae_in_train": true
  },
  "peak_gib": 34.38935327529907,
  "read_only": true
}
```
