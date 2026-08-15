#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/a4_protected.yaml
PYTHON=/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python
NPROC=4
MASTER_PORT=29641
PHASE="$1"

case "$PHASE" in
  analyze)
    "$PYTHON" -m torch.distributed.run --standalone --master-port "$MASTER_PORT" \
      --nproc-per-node "$NPROC" analyze_id_velocity.py --config "$CONFIG"
    ;;
  smoke)
    "$PYTHON" sampling_protected.py --config "$CONFIG" --device cuda:0 --smoke 2 --overwrite
    ;;
  tau-smoke)
    "$PYTHON" sampling_protected.py --config "$CONFIG" --device cuda:0 --mode tau_window \
      --smoke 2 --output-dir eval/a4_prot_tau_smoke --overwrite
    ;;
  main)
    "$PYTHON" -m torch.distributed.run --standalone --master-port "$MASTER_PORT" \
      --nproc-per-node "$NPROC" sampling_protected.py --config "$CONFIG" --mode main
    ;;
  main-metrics)
    "$PYTHON" eval_b2_metrics.py --config "$CONFIG" --subset eval/cf_subset.json --gen-dir eval/a4_prot_gen \
      --out-dir eval/a4_prot_metrics --report eval/a4_prot_metrics/report.md --metrics all --device cuda:0
    ;;
  tau-main|tau-ablation)
    "$PYTHON" -m torch.distributed.run --standalone --master-port "$MASTER_PORT" --nproc-per-node "$NPROC" sampling_protected.py \
      --config "$CONFIG" --mode tau_window
    ;;
  tau-metrics)
    "$PYTHON" eval_b2_metrics.py --config "$CONFIG" --subset eval/cf_subset.json \
      --gen-dir eval/a4_prot_tau_gen --out-dir eval/a4_prot_tau_metrics \
      --report eval/a4_prot_tau_metrics/report.md --metrics all --device cuda:0
    ;;
  scale03-ablation)
    "$PYTHON" -m torch.distributed.run --standalone --master-port "$MASTER_PORT" --nproc-per-node "$NPROC" sampling_protected.py \
      --config "$CONFIG" --mode scale03
    ;;
  scale03-metrics)
    "$PYTHON" eval_b2_metrics.py --config "$CONFIG" --subset eval/a4_prot_ablation_subset.json \
      --gen-dir eval/a4_prot_scale03_gen --out-dir eval/a4_prot_scale03_metrics \
      --report eval/a4_prot_scale03_metrics/report.md --metrics all --device cuda:0
    ;;
  gate)
    "$PYTHON" eval_protected_gate.py --config "$CONFIG"
    ;;
  *)
    printf '%s\n' \
      'usage: scripts/run_protected_inference.sh {analyze|smoke|tau-smoke|main|main-metrics|tau-main|tau-metrics|scale03-ablation|scale03-metrics|gate}'
    ;;
esac
