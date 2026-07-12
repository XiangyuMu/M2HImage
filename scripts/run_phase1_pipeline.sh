#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python

echo "[$(date '+%F %T %Z')] native-resolution cache coverage/build check"
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_DISABLE_XET=1   "$PY" -m torch.distributed.run --nproc_per_node=4   build_cache.py --config configs/warmup.yaml --split train,val,test

exec bash scripts/run_gatefix_to_b2.sh
