#!/usr/bin/env bash

# Append low-overhead health samples for the months-long full experiment.
# This process is intentionally observational: it never signals, restarts, or
# changes the queue or either GPU.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FULL_ENV="${M2H_FULL_ENV:-${PROJECT_ROOT}/configs/task_full_native_20260823.env}"
# shellcheck disable=SC1090
source "${FULL_ENV}"
source "${SCRIPT_DIR}/lib.sh"

queue_session="${M2H_FULL_QUEUE_SESSION:-m2h_full_native_queue}"
poll_seconds="${M2H_MONITOR_POLL_SECONDS:-900}"
monitor_root="${RUN_ROOT}/orchestrator"
health_tsv="${monitor_root}/health.tsv"
status_tsv="${monitor_root}/status.tsv"

case "${poll_seconds}" in
  ''|*[!0-9]*)
    printf 'M2H_MONITOR_POLL_SECONDS must be a positive integer, got %q\n' "${poll_seconds}" >&2
    exit 2
    ;;
esac
if [[ "${poll_seconds}" -lt 1 ]]; then
  printf 'M2H_MONITOR_POLL_SECONDS must be positive\n' >&2
  exit 2
fi

mkdir -p "${monitor_root}"
if [[ ! -e "${health_tsv}" ]]; then
  printf 'timestamp\tqueue_alive\tactive_log\tlog_bytes\tlog_mtime_epoch\tgpu0_temp_mem_util_power\tgpu1_temp_mem_util_power\tdata_avail_bytes\tstatus_rows\n' > "${health_tsv}"
fi

sample() {
  local timestamp queue_alive active_log log_bytes log_mtime data_avail status_rows
  local -a gpu_rows
  timestamp="$(date +%FT%T%z)"
  if tmux has-session -t "${queue_session}" 2>/dev/null; then
    queue_alive=1
  else
    queue_alive=0
  fi
  active_log="$(
    find "${monitor_root}/logs" -maxdepth 1 -type f -name '*_*.log' -printf '%T@\t%p\n' 2>/dev/null \
      | sort -nr | head -1 | cut -f2-
  )"
  if [[ -n "${active_log}" && -f "${active_log}" ]]; then
    log_bytes="$(stat -c %s "${active_log}")"
    log_mtime="$(stat -c %Y "${active_log}")"
  else
    active_log=""
    log_bytes=0
    log_mtime=0
  fi
  mapfile -t gpu_rows < <(
    nvidia-smi \
      --query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | tr -d ' '
  )
  data_avail="$(df -B1 --output=avail /data | tail -1 | tr -d ' ')"
  if [[ -f "${status_tsv}" ]]; then
    status_rows="$(( $(wc -l < "${status_tsv}") - 1 ))"
    if [[ "${status_rows}" -lt 0 ]]; then status_rows=0; fi
  else
    status_rows=0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${timestamp}" "${queue_alive}" "${active_log}" "${log_bytes}" "${log_mtime}" \
    "${gpu_rows[0]:-}" "${gpu_rows[1]:-}" "${data_avail}" "${status_rows}" \
    >> "${health_tsv}"
}

while :; do
  sample
  if [[ "${M2H_MONITOR_ONCE:-0}" == "1" ]]; then
    break
  fi
  if ! tmux has-session -t "${queue_session}" 2>/dev/null; then
    break
  fi
  sleep "${poll_seconds}"
done

printf 'Health history: %s\n' "${health_tsv}"
