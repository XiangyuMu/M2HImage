#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

method="${1:-all}"
profile="${2:-smoke}"
watch_mode="${3:-once}"
interval="${M2H_MONITOR_INTERVAL:-30}"
if [[ "${method}" != "all" ]]; then
  require_method "${method}"
fi
require_profile "${profile}"
if (( interval < 5 || interval > 60 )); then
  printf 'M2H_MONITOR_INTERVAL must be between 5 and 60 seconds.\n' >&2
  exit 2
fi

snapshot() {
  date '+%F %T %Z'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  ps -ef | rg 'train_refton_lora|image_train.py|train_xl.py|MCLD/train.py|train_m2h|infer_m2h|generate_vitonhd|IDM-VTON/inference.py' || true
  local search_root="${RUN_ROOT}"
  if [[ "${method}" != "all" ]]; then
    search_root="${RUN_ROOT}/${method}/${profile}"
  fi
  local latest_log
  latest_log="$(find "${search_root}" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1)"
  if [[ -n "${latest_log}" ]]; then
    printf 'Latest log: %s\n' "${latest_log#* }"
    tail -n 20 "${latest_log#* }"
  else
    printf 'No logs found under %s\n' "${search_root}"
  fi
}

snapshot
if [[ "${watch_mode}" == "watch" ]]; then
  while true; do
    sleep "${interval}"
    snapshot
  done
elif [[ "${watch_mode}" != "once" ]]; then
  printf 'Third argument must be once or watch.\n' >&2
  exit 2
fi
