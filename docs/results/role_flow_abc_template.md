# Role-Flow A/B/C Result Report Template

Use `tools/summarize_role_flow_experiments.py` after each run has a training
status, final checkpoint, and `evaluate_role_flow.py` summary. The generated
report records pending fields as `pending`; do not fill missing metrics by hand.

Required inputs per method:

- Config: `configs/role_selective/{A,B,C}_*.yaml`
- Run directory: `role_flow/runs/<run_id>/`
- Checkpoint: `role_flow/runs/<run_id>/checkpoints/final`
- Eval summary: `role_flow/eval/<run_id>/metrics/summary.json`
- Frozen eval manifest: `role_flow/eval_manifest.json`

Generated outputs:

- `role_flow_abc_summary.json`
- `role_flow_abc_summary.csv`
- `role_flow_abc_summary.md`

Metrics and directions:

| Metric | Direction |
| --- | --- |
| `id_cosine` | higher |
| `tar_at_1e-3` | higher |
| `garment_dino` | higher |
| `garment_iou` | higher |
| `pose_pck` | higher |
| `bg_ssim` | higher |
| `bg_lpips` | lower |
| `fid` | lower |

Default command:

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python tools/summarize_role_flow_experiments.py
```

Override paths explicitly when comparing reruns:

```bash
/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python tools/summarize_role_flow_experiments.py \
  --config A=configs/role_selective/A_timestep_routed.yaml \
  --run-dir A=/data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/runs/A_timestep_routed \
  --eval-summary A=/data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval/A_timestep_routed/metrics/summary.json \
  --checkpoint A=/data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/runs/A_timestep_routed/checkpoints/final
```
