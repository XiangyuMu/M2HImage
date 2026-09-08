#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FULL_ENV="${M2H_FULL_ENV:-${PROJECT_ROOT}/configs/task_full_native_20260823.env}"
# shellcheck disable=SC1090
source "${FULL_ENV}"
source "${SCRIPT_DIR}/lib.sh"

python="${M2H_PREP_PYTHON:-$(method_python refton)}"
manifest="${M2H_METRICS_MANIFEST:-${PREPARED_ROOT}/native/counterfactual/samples.jsonl}"
status_tsv="${RUN_ROOT}/orchestrator/status.tsv"
records="$(wc -l < "${manifest}")"
args=()
if [[ "${1:-}" == "--require-terminal" ]]; then
  args+=(--require-terminal)
elif [[ -n "${1:-}" ]]; then
  printf 'Usage: %s [--require-terminal]\n' "$0" >&2
  exit 2
fi

env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.summarize_full_native \
  --run-root "${RUN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --report-root "${REPORT_ROOT}" \
  --status-tsv "${status_tsv}" \
  --expected-records "${records}" \
  --native-manifest "${manifest}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  "${args[@]}"

printf 'Full comparison summary: %s/full_native_summary/report.md\n' "${REPORT_ROOT}"
