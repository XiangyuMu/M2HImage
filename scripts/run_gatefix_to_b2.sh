#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
ID=phase1_warmup_b2p_pulid_gatefix_resume_768x1024
OUT="$ROOT/phase1/$ID"
LOG_DIR="$OUT/logs"
CKPT_DIR="$OUT/checkpoints"
STOP_MARKER="$OUT/STOP_TRAINING"
RESUME_CKPT="$ROOT/phase1/phase1_warmup_b2p_pulid_gatefix_768x1024/checkpoints/final"
mkdir -p "$LOG_DIR" "$CKPT_DIR"
exec > >(tee -a "$LOG_DIR/gatefix_pipeline.log") 2>&1

if [[ -e "$STOP_MARKER" ]]; then
  echo "stale stop marker exists: $STOP_MARKER"
  exit 4
fi

echo "[$(date '+%F %T %Z')] watcher start on physical GPU3"
CUDA_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   "$PY" eval_watcher.py   --config configs/warmup.yaml   --ckpt-dir "$CKPT_DIR"   --device cuda:0 > "$LOG_DIR/watcher.log" 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > "$LOG_DIR/watcher.pid"
cleanup() {
  if kill -0 "$WATCHER_PID" 2>/dev/null; then
    kill "$WATCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T %Z')] 4400-step gatefix training start on GPU0-2"
set +e
CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   "$PY" -m torch.distributed.run --nproc_per_node=3   train_paired.py --config configs/warmup.yaml --resume "$RESUME_CKPT"
TRAIN_STATUS=$?
set -e
echo "[$(date '+%F %T %Z')] training exit status=$TRAIN_STATUS"

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  kill "$WATCHER_PID" 2>/dev/null || true
  wait "$WATCHER_PID" 2>/dev/null || true
  exit "$TRAIN_STATUS"
fi

echo "[$(date '+%F %T %Z')] waiting for final watcher report"
wait "$WATCHER_PID"

if [[ -e "$STOP_MARKER" ]]; then
  echo "watcher hard gate failed; B2 generation will not run"
  exit 3
fi

echo "[$(date '+%F %T %Z')] watcher passed; starting B2 generation"
bash scripts/run_b2_generation.sh "$CKPT_DIR/final"

echo "[$(date '+%F %T %Z')] B2 official metrics start"
CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py   --config configs/warmup.yaml   --subset "$ROOT/eval/cf_subset.json"   --gen-dir "$ROOT/eval/b2p_gatefix_gen"   --metrics all   --device cuda:0

echo "[$(date '+%F %T %Z')] gatefix training + B2 pipeline complete"
