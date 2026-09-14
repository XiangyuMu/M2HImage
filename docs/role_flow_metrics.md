# Role-flow evaluation protocol

`tools/evaluate_role_flow.py` evaluates A, B, and C with the same explicit
pair manifest. The manifest must contain `mid`, `jid`, `seed`, and `split`
(`val` or `test`). Optional absolute or dataset-relative paths are
`generated_path`, `mannequin_path`, and `human_path`; omitted paths are derived
from the configured dataset and generated directory.

The official held-out AdaFace checkpoint is used only by this evaluator.
Identity cosine is the generated image embedding cosine against the target
human identity reference. `TAR@FAR=1e-3` is calibrated from validation
genuine scores and generated-to-wrong-reference impostor scores. The resulting
threshold is frozen while evaluating the held-out test split. A run fails
closed when the AdaFace weights, detector, or face references are unavailable.

Garment DINO uses the existing GSVTON DINOv2 ViT-B/14 implementation and the
mannequin cloth mask. Garment IoU compares that mask with the generated image's
FASHN parsing garment union. Pose PCK uses generated DWPose body keypoints
against the mannequin keypoints, requiring confidence `>=0.3` and counting a
correct point when normalized distance is `<=0.05`.

Background SSIM and LPIPS are computed on the common background region
(`not generated garment OR mannequin garment`); pixels outside that region are
set to gray in both images. A sample with no usable common background is a
metric failure. FID is an actual torch-fidelity Inception FID against the
unique target human references in the same manifest. It is a set-level value
and is repeated in each per-pair row for convenient joins.

Example:

```bash
python tools/evaluate_role_flow.py \
  --config configs/role_selective/A_timestep_routed.yaml \
  --manifest /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval_manifest.json \
  --generated-dir /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval/A_timestep_routed/generated \
  --output-dir /data/muxiangyu/experiments/M2H_Final_v2_clean_v1/role_flow/eval/A_timestep_routed/metrics_final \
  --split test --calibration-split val --device cuda:0
```

Outputs are `per_pair_metrics.csv`, `failures.csv`, and `summary.json`.
`summary.json` records the manifest, config, checkpoint hashes, calibration
threshold, per-metric valid counts, and failure counts. Reuse exactly the
same manifest for A/B/C; the manifest SHA-256 in each summary is the protocol
identity. Use validation outputs for development only and report the held-out
test summary for the final comparison.
