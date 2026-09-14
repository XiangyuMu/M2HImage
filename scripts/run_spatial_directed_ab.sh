#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
SUBSET="$ROOT/eval/cf_subset.json"
METRICS_CONFIG=configs/spatial_hair_ab_metrics_v2.yaml
DECISION="$ROOT/eval/spatial_hair_ab_decision.json"
STAGE="$1"
TRAIN_PID=""
WATCHER_PID=""

cleanup_children() {
  for pid in "$WATCHER_PID" "$TRAIN_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup_children EXIT

winner_branch() {
  test -f "$DECISION"
  jq -r '.winner' "$DECISION"
}

branch_config() {
  local kind="$1"
  local winner="$2"
  echo "configs/spatial_c_$kind"_"$winner.yaml"
}

branch_run_id() {
  local kind="$1"
  local winner="$2"
  echo "phase1_spatial_c_$kind"_"$winner"_r16_12400_768x1024
}

run_smoke() {
  local kind="$1"
  local winner="$2"
  local config smoke_id smoke_dir
  config="$(branch_config "$kind" "$winner")"
  smoke_id="smoke20_spatial_c_$kind"_"$winner"_20260825
  smoke_dir="$ROOT/phase1/$smoke_id"
  if [[ -f "$smoke_dir/training_status.json" ]] &&
     grep -q '"status": "complete"' "$smoke_dir/training_status.json"; then
    echo "smoke already complete: $smoke_dir"
    return
  fi
  if [[ -d "$smoke_dir" ]] && [[ -n "$(ls -A "$smoke_dir")" ]]; then
    echo "incomplete smoke directory requires inspection: $smoke_dir" >&2
    return 5
  fi
  CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     "$PY" train_paired.py --config "$config" --dev-single-gpu       --smoke-steps 20 --override-output-id "$smoke_id"
}

run_train() {
  local kind="$1"
  local winner="$2"
  local config run_id run_dir log_dir ckpt_dir
  config="$(branch_config "$kind" "$winner")"
  run_id="$(branch_run_id "$kind" "$winner")"
  run_dir="$ROOT/phase1/$run_id"
  log_dir="$run_dir/logs"
  ckpt_dir="$run_dir/checkpoints"
  if [[ -f "$run_dir/training_status.json" ]] &&
     grep -q '"status": "complete"' "$run_dir/training_status.json" &&
     [[ -f "$ckpt_dir/final/READY" ]]; then
    echo "training already complete: $run_dir"
    return
  fi
  if [[ -d "$run_dir" ]] && [[ -n "$(ls -A "$run_dir")" ]]; then
    echo "formal run directory is non-empty but incomplete: $run_dir" >&2
    return 5
  fi
  mkdir -p "$log_dir" "$ckpt_dir"

  CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_DISABLE_XET=1   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     "$PY" -m torch.distributed.run --standalone --nproc_per_node=3       train_paired.py --config "$config"       >"$log_dir/train.launch.log" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" >"$log_dir/train.pid"

  CUDA_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     "$PY" eval_watcher.py --config "$config" --ckpt-dir "$ckpt_dir"       --device cuda:0 >"$log_dir/watcher.log" 2>&1 &
  WATCHER_PID=$!
  echo "$WATCHER_PID" >"$log_dir/watcher.pid"

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if ! kill -0 "$WATCHER_PID" 2>/dev/null; then
      set +e
      wait "$WATCHER_PID"
      local watcher_status=$?
      set -e
      WATCHER_PID=""
      if [[ "$watcher_status" -ne 0 || ! -f "$ckpt_dir/final/READY" ]]; then
        kill "$TRAIN_PID" 2>/dev/null || true
        echo "watcher exited before accepted final checkpoint: $kind/$winner" >&2
        return 6
      fi
      break
    fi
    sleep 30
  done

  set +e
  wait "$TRAIN_PID"
  local train_status=$?
  set -e
  TRAIN_PID=""
  if [[ "$train_status" -ne 0 ]]; then
    kill "$WATCHER_PID" 2>/dev/null || true
    return "$train_status"
  fi
  if [[ -n "$WATCHER_PID" ]]; then
    wait "$WATCHER_PID"
  fi
  if [[ -e "$run_dir/STOP_TRAINING" ]]; then
    echo "watcher rejected branch $kind/$winner" >&2
    return 3
  fi
}

evaluate_branch() {
  local kind="$1"
  local winner="$2"
  local config run_id run_dir final metric_name gen_dir legacy
  config="$(branch_config "$kind" "$winner")"
  run_id="$(branch_run_id "$kind" "$winner")"
  run_dir="$ROOT/phase1/$run_id"
  final="$run_dir/checkpoints/final"
  metric_name="spatial_c_$kind"_"$winner"
  gen_dir="$ROOT/eval/$metric_name"_gen
  legacy="$ROOT/eval/$metric_name"_legacy_metrics
  test -f "$final/READY"
  test ! -e "$run_dir/STOP_TRAINING"

  CUDA_VISIBLE_DEVICES=0,1,2,3     "$PY" -m torch.distributed.run --standalone --nproc_per_node=4       eval_b2.py --config "$config" --ckpt "$final" --subset "$SUBSET"

  CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py     --config "$config" --subset "$SUBSET" --gen-dir "$gen_dir"     --metrics all --device cuda:0 --out-dir "$legacy"     --report "$ROOT/eval/$metric_name"_legacy_report.md

  CUDA_VISIBLE_DEVICES=0,1 "$PY" eval_metrics_v2.py     --config "$METRICS_CONFIG" --run "$metric_name" --metrics all     --device cuda:0 --pose-device cuda:1
}

write_report() {
  "$PY" eval_spatial_directed_gate_report.py
}

WINNER="$(winner_branch)"
case "$WINNER" in space|semantic) ;; *) echo "invalid winner: $WINNER" >&2; exit 5 ;; esac
test -f "artifacts/hair_ab_fixed_set/$WINNER/step8400.json"

case "$STAGE" in
  smoke)
    run_smoke a4prime "$WINNER"
    run_smoke cont "$WINNER"
    ;;
  train)
    run_train a4prime "$WINNER"
    run_train cont "$WINNER"
    ;;
  evaluate)
    evaluate_branch a4prime "$WINNER"
    evaluate_branch cont "$WINNER"
    ;;
  report)
    write_report
    ;;
  all)
    run_smoke a4prime "$WINNER"
    run_smoke cont "$WINNER"
    run_train a4prime "$WINNER"
    run_train cont "$WINNER"
    evaluate_branch a4prime "$WINNER"
    evaluate_branch cont "$WINNER"
    write_report
    ;;
  *)
    echo "usage: $0 {smoke|train|evaluate|report|all}" >&2
    exit 2
    ;;
esac
