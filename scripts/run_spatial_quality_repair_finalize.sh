#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
CONFIG=configs/spatial_quality_repair_resume.yaml
RUN="$ROOT/phase1/phase1_spatial_quality_repair_r16_6400_768x1024"
FINAL="$RUN/checkpoints/final"
SUBSET="$ROOT/eval/cf_subset.json"
GEN="$ROOT/eval/spatial_quality_repair_gen"
LEGACY="$ROOT/eval/spatial_quality_repair_legacy_metrics"

if [[ ! -f "$FINAL/READY" ]]; then
  echo "quality-repair final checkpoint is not READY: $FINAL" >&2
  exit 5
fi
if [[ -f "$RUN/STOP_TRAINING" ]]; then
  echo "quality-repair run is rejected: $RUN/STOP_TRAINING" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 eval_b2.py \
  --config "$CONFIG" --ckpt "$FINAL" --subset "$SUBSET"

CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py \
  --config "$CONFIG" --subset "$SUBSET" --gen-dir "$GEN" \
  --metrics all --device cuda:0 --out-dir "$LEGACY" \
  --report "$ROOT/eval/spatial_quality_repair_legacy_report.md"

CUDA_VISIBLE_DEVICES=0,1 "$PY" eval_metrics_v2.py \
  --config configs/spatial_metrics_v2.yaml \
  --run spatial_quality_repair \
  --compare spatial_quality_repair spatial_hair_incontext_fixedset \
  --metrics all --device cuda:0 --pose-device cuda:1

CUDA_VISIBLE_DEVICES=0 "$PY" eval_metrics_v2.py \
  --config configs/spatial_metrics_v2.yaml \
  --compare-only --compare spatial_quality_repair a4 \
  --device cuda:0 --pose-device cuda:1

printf 'complete\n' >"$RUN/FINALIZE_COMPLETE"
