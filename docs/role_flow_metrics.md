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

## Protocol v2

`tools/evaluate_role_flow_v2.py` is the paper-facing evaluator. It freezes the
full 580-pair generation set before any metric model is loaded: 180 validation
pairs and 400 test pairs must be present, the generated directory must contain
exactly the expected PNG filenames, every generated PNG must be readable RGB at
the configured resolution, and `generation_provenance.json` must match the
config, checkpoint, manifest, row keys, image hashes, and complete status.

V2 loads the held-out AdaFace `tar_calibration.json` threshold as an input
artifact and does not recalibrate TAR during evaluation. The full 580-row set
is used for generation preflight only; all per-pair metrics and formal means
are computed on the 400-row final `test` split. The 180 `val` rows must still
be generated and readable, but are excluded from metric aggregation. FID is a
set-level metric only: generated images are the 400 final-test outputs and
the real distribution is the frozen 1,971-row final-test human reference
manifest, while `per_pair_metrics.csv` intentionally contains no FID column.

Garment source masks come only from mannequin-side FASHN parsing labels
`3,4,5,6,7,10`. Generated garment masks come from generated-image FASHN parsing;
if the generated garment area is below the fixed threshold, garment metrics are
marked invalid rather than falling back to a projected source mask. Background
metrics use the full non-background foreground union from source and generated
FASHN labels, erode the common background with an 11x11 kernel, then compute
masked SSIM and spatial LPIPS.

Each metric row includes a stable JSON `metric_provenance` field identifying
the mannequin parsing source, generated parsing source, and background-mask
construction. Outputs are `per_pair_metrics.csv`, `failures.csv`,
`set_metrics.json`, `summary.json`, `provenance.json`, and `READY`. The output
directory must be empty at start so stale metrics cannot be silently reused.
