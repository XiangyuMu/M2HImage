#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
RUN="$ROOT/phase1/phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024"
FINAL="$RUN/checkpoints/final"
CONFIG=configs/spatial_warmup_resume_hair_incontext.yaml
SUBSET="$ROOT/eval/cf_subset.json"
GEN="$ROOT/eval/b2p_spatial_hair_incontext_fixedset_gen"
METRICS="$ROOT/eval/b2p_spatial_hair_incontext_fixedset_metrics"

while [[ ! -f "$FINAL/READY" ]]; do
  if [[ -f "$RUN/STOP_TRAINING" ]]; then
    echo "finalizer: training stopped before final; see $RUN/STOP_TRAINING" >&2
    exit 3
  fi
  sleep 300
done

while [[ ! -f "$RUN/warmup_vis/final/watcher_report.md" ]]; do
  if [[ -f "$RUN/STOP_TRAINING" ]]; then
    echo "finalizer: final watcher rejected the run; see $RUN/STOP_TRAINING" >&2
    exit 3
  fi
  sleep 60
done

if [[ -f "$RUN/STOP_TRAINING" ]]; then
  echo "finalizer: refusing evaluation while STOP_TRAINING exists" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 eval_b2.py \
  --config "$CONFIG" --ckpt "$FINAL" --subset "$SUBSET"

CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py \
  --config "$CONFIG" --subset "$SUBSET" --gen-dir "$GEN" \
  --metrics all --device cuda:0 --out-dir "$METRICS" \
  --report "$ROOT/eval/b2p_spatial_hair_incontext_fixedset_report.md"

CUDA_VISIBLE_DEVICES=0,1 "$PY" eval_metrics_v2.py \
  --config configs/spatial_metrics_v2.yaml \
  --run spatial_hair_incontext_fixedset \
  --compare spatial_hair_incontext_fixedset a4 \
  --metrics all --device cuda:0 --pose-device cuda:1

printf 'complete\n' >"$RUN/FINALIZE_COMPLETE"
