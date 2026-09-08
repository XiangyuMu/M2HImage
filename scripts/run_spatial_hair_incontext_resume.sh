#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
CONFIG=configs/spatial_warmup_resume_hair_incontext.yaml
RUN_ID=phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024
RUN_DIR="$ROOT/phase1/$RUN_ID"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
STOP_MARKER="$RUN_DIR/STOP_TRAINING"
RESUME_CKPT="$ROOT/phase1/phase1_spatial_hair_incontext_resume_v2_r16_4400_768x1024/checkpoints/step-002500"
BASELINE_2000=artifacts/rebaseline_fixed_set/step2000.json
BASELINE_2500=artifacts/rebaseline_fixed_set/step2500.json

mkdir -p "$CKPT_DIR" "$LOG_DIR"
if [[ -e "$STOP_MARKER" ]]; then
  echo "stale stop marker exists: $STOP_MARKER" >&2
  echo "inspect it before resuming; this launcher will not remove stop markers" >&2
  exit 4
fi

cleanup() {
  for pid in "${WATCHER_PID:-}" "${TRAIN_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

WATCHER_PID=
TRAIN_PID=
if [[ ! -f "$RESUME_CKPT/READY" ]]; then
  echo "resume checkpoint is not READY: $RESUME_CKPT" >&2
  exit 5
fi
if [[ ! -f "$BASELINE_2000" || ! -f "$BASELINE_2500" ]]; then
  echo "fixed-set historical snapshots are missing; run rebaseline_watcher first" >&2
  exit 5
fi
"$PY" tools/rebaseline_watcher.py \
  --config "$CONFIG" \
  --output artifacts/rebaseline_fixed_set \
  --report-only >"$LOG_DIR/rebaseline_preflight.log" 2>&1

echo "[$(date '+%F %T %Z')] step-2500 fixed-set continuation starting on physical GPU0-2"
CUDA_VISIBLE_DEVICES=0,1,2 \
HF_HUB_DISABLE_XET=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  train_paired.py --config "$CONFIG" >"$LOG_DIR/train.launch.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" >"$LOG_DIR/train.pid"

echo "[$(date '+%F %T %Z')] checkpoint watcher starting on physical GPU3"
CUDA_VISIBLE_DEVICES=3 \
HF_HUB_DISABLE_XET=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" eval_watcher.py \
  --config "$CONFIG" \
  --ckpt-dir "$CKPT_DIR" \
  --device cuda:0 >"$LOG_DIR/watcher.log" 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" >"$LOG_DIR/watcher.pid"

WATCHER_STATUS=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
    set +e
    wait "$WATCHER_PID"
    WATCHER_STATUS=$?
    set -e
    WATCHER_PID=
    if [[ "$WATCHER_STATUS" -ne 0 || ! -f "$CKPT_DIR/final/READY" ]]; then
      echo "checkpoint watcher exited before a valid final checkpoint; stopping training" >&2
      exit 6
    fi
    break
  fi
  sleep 30
done

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
TRAIN_PID=

echo "[$(date '+%F %T %Z')] training exit status=$TRAIN_STATUS"
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi

echo "[$(date '+%F %T %Z')] waiting for final watcher report"
if [[ -n "$WATCHER_PID" ]]; then
  wait "$WATCHER_PID"
  WATCHER_PID=
fi

if [[ -e "$STOP_MARKER" ]]; then
  echo "watcher rejected the continuation; inspect $STOP_MARKER" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] continuation and final watcher completed"
