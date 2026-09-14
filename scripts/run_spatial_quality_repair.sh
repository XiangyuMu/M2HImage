#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
CONFIG=configs/spatial_quality_repair_resume.yaml
RUN_ID=phase1_spatial_quality_repair_r16_6400_768x1024
RUN_DIR="$ROOT/phase1/$RUN_ID"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
STOP_MARKER="$RUN_DIR/STOP_TRAINING"
RESUME_CKPT="$ROOT/phase1/phase1_spatial_hair_incontext_fixedset_resume_r16_4400_768x1024/checkpoints/final"
BASELINE=artifacts/rebaseline_fixed_set/step4400.json
AUDIT=artifacts/ref_audit_quality_repair/report.md
SMOKE="$ROOT/phase1/spatial_quality_repair_smoke20_bf16_scale048/training_status.json"
VRAM=docs/results/vram_report_spatial_quality_repair_r16_768x1024.md

if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "formal run directory is not clean: $RUN_DIR" >&2
  exit 5
fi
mkdir -p "$CKPT_DIR" "$LOG_DIR"
for required in "$RESUME_CKPT/READY" "$BASELINE" "$AUDIT" "$SMOKE" "$VRAM"; do
  if [[ ! -f "$required" ]]; then
    echo "quality-repair preflight is missing: $required" >&2
    exit 5
  fi
done
if ! grep -q 'Conclusion: NO-LEAK' "$AUDIT"; then
  echo "reference audit did not pass NO-LEAK: $AUDIT" >&2
  exit 5
fi
if ! grep -q '"checkpoint_step": 4400' "$BASELINE"; then
  echo "fixed-set baseline is not the immutable step-4400 checkpoint: $BASELINE" >&2
  exit 5
fi
if ! grep -q '"status": "complete"' "$SMOKE"; then
  echo "20-step smoke is not complete: $SMOKE" >&2
  exit 5
fi
if ! grep -q 'PASS.*44' "$VRAM"; then
  echo "VRAM report has no <=44 GiB PASS: $VRAM" >&2
  exit 5
fi
if [[ -e "$STOP_MARKER" ]]; then
  echo "stale stop marker exists: $STOP_MARKER" >&2
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

echo "[$(date '+%F %T %Z')] quality-repair continuation 4400->6400 on GPU0-2"
CUDA_VISIBLE_DEVICES=0,1,2 \
HF_HUB_DISABLE_XET=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 \
  train_paired.py --config "$CONFIG" >"$LOG_DIR/train.launch.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" >"$LOG_DIR/train.pid"

echo "[$(date '+%F %T %Z')] fixed-set watcher on GPU3"
CUDA_VISIBLE_DEVICES=3 \
HF_HUB_DISABLE_XET=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" eval_watcher.py --config "$CONFIG" --ckpt-dir "$CKPT_DIR" \
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
      echo "watcher exited before an accepted final checkpoint" >&2
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
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi
if [[ -n "$WATCHER_PID" ]]; then
  wait "$WATCHER_PID"
  WATCHER_PID=
fi
if [[ -e "$STOP_MARKER" ]]; then
  echo "watcher rejected the run: $STOP_MARKER" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] quality-repair continuation accepted"
