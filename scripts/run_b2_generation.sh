#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
ID=phase1_warmup_b2p_pulid_gatefix_resume_768x1024
LOG_DIR="$ROOT/phase1/$ID/logs"
CKPT="$ROOT/phase1/$ID/checkpoints/final"
if [[ "$#" -gt 0 ]]; then
  CKPT="$1"
fi
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/b2_generation.log") 2>&1

echo "[$(date '+%F %T %Z')] B2 generation start ckpt=$CKPT"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --nproc_per_node=4   eval_b2.py   --config configs/warmup.yaml   --ckpt "$CKPT"   --subset "$ROOT/eval/cf_subset.json"
echo "[$(date '+%F %T %Z')] B2 generation done"
find "$ROOT/eval/b2p_gatefix_gen" -maxdepth 1 -type f -name '*.png' | wc -l
