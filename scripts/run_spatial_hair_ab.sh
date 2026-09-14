#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

PY=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
ROOT=/data/muxiangyu/datasets/M2HImage/M2H_Final_v2
SUBSET="$ROOT/eval/cf_subset.json"
METRICS_CONFIG=configs/spatial_hair_ab_metrics_v2.yaml
CURRENT_CKPT="$ROOT/phase1/phase1_spatial_quality_repair_r16_6400_768x1024/checkpoints/final"
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

branch_config() {
  case "$1" in
    space) echo configs/spatial_hair_ab_space.yaml ;;
    semantic) echo configs/spatial_hair_ab_semantic.yaml ;;
    *) echo "unknown Part B branch: $1" >&2; return 2 ;;
  esac
}

branch_run_id() {
  case "$1" in
    space) echo phase1_spatial_hair_ab_space_r16_8400_768x1024 ;;
    semantic) echo phase1_spatial_hair_ab_semantic_r16_8400_768x1024 ;;
  esac
}

preflight() {
  test -f "$CURRENT_CKPT/READY"
  test -f "$SUBSET"
  test -f artifacts/id_failure_attrib/report.md
  test -f configs/watcher_eval_set.json
  test "$(rg --files "$ROOT/phase1/cache_spatial_quality_repair_768x1024/samples" | wc -l)" -eq 40014
}

run_smoke() {
  local branch="$1"
  local config
  config="$(branch_config "$branch")"
  local smoke_id="smoke20_spatial_hair_ab_$branch"_20260825
  local smoke_dir="$ROOT/phase1/$smoke_id"
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

build_current_fixed_baseline() {
  local output=artifacts/hair_ab_fixed_set/current
  if [[ -f "$output/step6400.json" ]]; then
    echo "fixed-set baseline already exists: $output/step6400.json"
    return
  fi
  CUDA_VISIBLE_DEVICES=3 HF_HUB_DISABLE_XET=1   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     "$PY" tools/rebaseline_watcher.py       --config configs/spatial_hair_ab_space.yaml --ckpt "$CURRENT_CKPT"       --device cuda:0 --output "$output"
}

run_train() {
  local branch="$1"
  local config run_id run_dir log_dir ckpt_dir
  config="$(branch_config "$branch")"
  run_id="$(branch_run_id "$branch")"
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
        echo "watcher exited before accepted final checkpoint: $branch" >&2
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
    return "$train_status"
  fi
  if [[ -n "$WATCHER_PID" ]]; then
    wait "$WATCHER_PID"
    WATCHER_PID=""
  fi
  if [[ -e "$run_dir/STOP_TRAINING" ]]; then
    echo "watcher rejected branch $branch: $run_dir/STOP_TRAINING" >&2
    return 3
  fi
}

evaluate_branch() {
  local branch="$1"
  local config run_id run_dir final metric_name gen_dir legacy attrib fixed
  config="$(branch_config "$branch")"
  run_id="$(branch_run_id "$branch")"
  run_dir="$ROOT/phase1/$run_id"
  final="$run_dir/checkpoints/final"
  metric_name="spatial_hair_ab_$branch"
  gen_dir="$ROOT/eval/$metric_name"_gen
  legacy="$ROOT/eval/$metric_name"_legacy_metrics
  attrib="artifacts/id_failure_attrib/hair_ab_$branch"
  fixed="artifacts/hair_ab_fixed_set/$branch"
  test -f "$final/READY"
  test ! -e "$run_dir/STOP_TRAINING"

  CUDA_VISIBLE_DEVICES=0,1,2,3     "$PY" -m torch.distributed.run --standalone --nproc_per_node=4       eval_b2.py --config "$config" --ckpt "$final" --subset "$SUBSET"

  CUDA_VISIBLE_DEVICES=0 "$PY" eval_b2_metrics.py     --config "$config" --subset "$SUBSET" --gen-dir "$gen_dir"     --metrics all --device cuda:0 --out-dir "$legacy"     --report "$ROOT/eval/$metric_name"_legacy_report.md

  CUDA_VISIBLE_DEVICES=0,1 "$PY" eval_metrics_v2.py     --config "$METRICS_CONFIG" --run "$metric_name" --metrics all     --device cuda:0 --pose-device cuda:1

  "$PY" tools/attribute_identity_failures.py     --config "$METRICS_CONFIG" --run "$metric_name" --output-dir "$attrib"

  CUDA_VISIBLE_DEVICES=3 "$PY" tools/rebaseline_watcher.py     --config "$config" --ckpt "$final" --device cuda:0 --output "$fixed"
}

write_report() {
  "$PY" eval_hair_ab_report.py
}

case "$STAGE" in
  smoke)
    preflight
    run_smoke space
    run_smoke semantic
    ;;
  baseline)
    preflight
    build_current_fixed_baseline
    ;;
  train)
    preflight
    build_current_fixed_baseline
    run_train space
    run_train semantic
    ;;
  evaluate)
    evaluate_branch space
    evaluate_branch semantic
    ;;
  report)
    write_report
    ;;
  all)
    preflight
    run_smoke space
    run_smoke semantic
    build_current_fixed_baseline
    run_train space
    run_train semantic
    evaluate_branch space
    evaluate_branch semantic
    write_report
    ;;
  *)
    echo "usage: $0 {smoke|baseline|train|evaluate|report|all}" >&2
    exit 2
    ;;
esac
