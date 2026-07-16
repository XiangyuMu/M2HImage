#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
CONFIG=configs/a4_directed.yaml
RUN_ID=phase2_a4_directed_r16_4000_768x1024
RUN_DIR="$ROOT/phase1/$RUN_ID"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
MODE="${1:-all}"

test -f "$ROOT/derived/identity_bank_v2.npz"
test -f "$ROOT/derived/region_masks_z/manifest.json"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

run_train() {
  if [[ -f "$CKPT_DIR/final/READY" ]]; then
    echo "A4 final checkpoint already exists; refusing a second mechanism run. Use --eval-only."
    exit 5
  fi
  if [[ -e "$RUN_DIR/STOP_TRAINING" ]]; then
    echo "stale STOP_TRAINING marker: $RUN_DIR/STOP_TRAINING"
    exit 4
  fi

  echo "[$(date '+%F %T %Z')] A4 watcher start"
  CUDA_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" eval_watcher.py --config "$CONFIG" --ckpt-dir "$CKPT_DIR" --device cuda:0 \
    > "$LOG_DIR/watcher.log" 2>&1 &
  watcher_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --nproc_per_node=3 \
    train_paired.py --config "$CONFIG"
  train_status=$?
  set -e
  if [[ "$train_status" -ne 0 ]]; then
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
    return "$train_status"
  fi
  wait "$watcher_pid"
}

run_eval() {
  test -f "$CKPT_DIR/final/READY"
  echo "[$(date '+%F %T %Z')] A4 generation start"
  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --nproc_per_node=4 \
    eval_b2.py --config "$CONFIG" --ckpt "$CKPT_DIR/final" --subset "$ROOT/eval/cf_subset.json"

  echo "[$(date '+%F %T %Z')] A4 frozen metrics start"
  CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py \
    --config "$CONFIG" \
    --subset "$ROOT/eval/cf_subset.json" \
    --gen-dir "$ROOT/eval/a4_gen" \
    --metrics all \
    --device cuda:0 \
    --out-dir "$ROOT/eval/a4_metrics" \
    --report "$ROOT/eval/a4_report.md"

  "$PY" eval_a4_gate_report.py \
    --a4-config "$CONFIG" \
    --b2cont-config configs/b2_cont.yaml \
    --a4-metrics eval/a4_metrics \
    --b2cont-metrics eval/b2cont_metrics \
    --a2-metrics eval/a2_metrics \
    --report eval/a4_gate_report.md
}

case "$MODE" in
  all)
    run_train
    run_eval
    ;;
  --eval-only)
    run_eval
    ;;
  *)
    echo "usage: $0 [all|--eval-only]" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T %Z')] A4 one-shot gate complete"
