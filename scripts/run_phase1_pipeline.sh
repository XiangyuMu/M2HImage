#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
LOG_DIR=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_warmup_b2/logs
CKPT_DIR=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2/phase1/phase1_warmup_b2/checkpoints
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

echo "[$(date '+%F %T %Z')] phase1 pipeline start"
echo "[$(date '+%F %T %Z')] build cache start: train,val,test on GPU0-3"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --nproc_per_node=4 build_cache.py --config configs/warmup.yaml --split train,val,test

echo "[$(date '+%F %T %Z')] build cache complete"
echo "[$(date '+%F %T %Z')] watcher start on physical GPU3"
CUDA_VISIBLE_DEVICES=3 "$PY" eval_watcher.py --config configs/warmup.yaml --ckpt-dir "$CKPT_DIR" --device cuda:0 > "$LOG_DIR/watcher.log" 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > "$LOG_DIR/watcher.pid"
cleanup() {
  if [[ -n "${WATCHER_PID:-}" ]]; then
    kill "$WATCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T %Z')] training start: 3-rank DDP on physical GPU0-2"
set +e
CUDA_VISIBLE_DEVICES=0,1,2 "$PY" -m torch.distributed.run --nproc_per_node=3 train_paired.py --config configs/warmup.yaml
STATUS=$?
set -e
echo "[$(date '+%F %T %Z')] training exit status=$STATUS"
exit "$STATUS"
