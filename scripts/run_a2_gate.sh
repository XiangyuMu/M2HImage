#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage
PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2

run_train() {
  local config="$1"
  local run_id="$2"
  local out="$ROOT/phase1/$run_id"
  local ckpt_dir="$out/checkpoints"
  local log_dir="$out/logs"
  mkdir -p "$log_dir" "$ckpt_dir"
  if [[ -e "$out/STOP_TRAINING" ]]; then
    echo "stale STOP_TRAINING marker: $out/STOP_TRAINING"
    exit 4
  fi

  echo "[$(date '+%F %T %Z')] watcher start config=$config"
  CUDA_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" eval_watcher.py --config "$config" --ckpt-dir "$ckpt_dir" --device cuda:0 \
    > "$log_dir/watcher.log" 2>&1 &
  local watcher_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --nproc_per_node=3 \
    train_paired.py --config "$config"
  local train_status=$?
  set -e
  if [[ "$train_status" -ne 0 ]]; then
    kill "$watcher_pid" 2>/dev/null || true
    wait "$watcher_pid" 2>/dev/null || true
    return "$train_status"
  fi
  wait "$watcher_pid"
}

run_eval() {
  local config="$1"
  local run_id="$2"
  local gen_dir="$3"
  local metrics_dir="$4"
  local report="$5"
  local ckpt="$ROOT/phase1/$run_id/checkpoints/final"

  echo "[$(date '+%F %T %Z')] generation start config=$config"
  CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --nproc_per_node=4 \
    eval_b2.py --config "$config" --ckpt "$ckpt" --subset "$ROOT/eval/cf_subset.json"

  echo "[$(date '+%F %T %Z')] metrics start gen=$gen_dir"
  CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py \
    --config "$config" \
    --subset "$ROOT/eval/cf_subset.json" \
    --gen-dir "$ROOT/$gen_dir" \
    --metrics all \
    --device cuda:0 \
    --out-dir "$ROOT/$metrics_dir" \
    --report "$ROOT/$report"
}

test -f "$ROOT/derived/identity_bank.npz"
test -f "$ROOT/derived/region_masks_z/manifest.json"

run_train configs/a2_diff.yaml phase2_a2_diff_r16_4000_768x1024
run_train configs/b2_cont.yaml phase2_b2_cont_r16_4000_768x1024

run_eval configs/a2_diff.yaml phase2_a2_diff_r16_4000_768x1024 eval/a2_gen eval/a2_metrics eval/a2_report.md
run_eval configs/b2_cont.yaml phase2_b2_cont_r16_4000_768x1024 eval/b2cont_gen eval/b2cont_metrics eval/b2cont_report.md

"$PY" eval_gate_report.py \
  --a2-config configs/a2_diff.yaml \
  --b2cont-config configs/b2_cont.yaml \
  --a2-metrics eval/a2_metrics \
  --b2cont-metrics eval/b2cont_metrics \
  --b2p-metrics eval/b2p_gatefix_metrics \
  --report eval/gate_report.md

echo "[$(date '+%F %T %Z')] A2 gate experiment complete"
