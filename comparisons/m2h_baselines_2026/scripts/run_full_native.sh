#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FULL_ENV="${M2H_FULL_ENV:-${PROJECT_ROOT}/configs/task_full_native_20260823.env}"
if [[ -f "${FULL_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${FULL_ENV}"
fi
source "${SCRIPT_DIR}/lib.sh"

PREP_PYTHON="${M2H_PREP_PYTHON:-$(method_python refton)}"
NATIVE_MANIFEST="${M2H_METRICS_MANIFEST:-${PREPARED_ROOT}/native/counterfactual/samples.jsonl}"
LOW_MANIFEST="${PREPARED_ROOT}/low/counterfactual/samples.jsonl"
STATUS_ROOT="${RUN_ROOT}/orchestrator"
STATUS_TSV="${STATUS_ROOT}/status.tsv"
OOM_RC=125
OOM_PATTERN='out[[:space:]_-]*of[[:space:]_-]*memory|cublas_status_alloc_failed|cuda error.*memory|memory allocation failed|cannot allocate memory'

prepare=1
prepare_only=0
case "${1:-}" in
  --skip-prepare) prepare=0 ;;
  --prepare-only) prepare_only=1 ;;
  "") ;;
  *)
    printf 'Usage: %s [--skip-prepare|--prepare-only]\n' "$0" >&2
    exit 2
    ;;
esac

mkdir -p "${STATUS_ROOT}/logs"
if [[ -e "${STATUS_TSV}" ]]; then
  printf 'Refusing to reuse existing full queue status: %s\n' "${STATUS_TSV}" >&2
  printf 'Choose a new M2H_RUN_ROOT or move the previous run aside explicitly.\n' >&2
  exit 2
fi
printf 'method\tphase\tstatus\trc\tlog\n' > "${STATUS_TSV}"

record() {
  local method="$1" phase="$2" status="$3" rc="$4" log="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "${method}" "${phase}" "${status}" "${rc}" "${log}" >> "${STATUS_TSV}"
}

has_oom() {
  local log="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "${OOM_PATTERN}" "${log}"
  else
    grep -Eiq "${OOM_PATTERN}" "${log}"
  fi
}

run_phase() {
  local method="$1" phase="$2"
  shift 2
  local command="$1"
  shift
  local log="${STATUS_ROOT}/logs/${method}_${phase}_$(date +%Y%m%d-%H%M%S).log"
  printf '\n[%s/%s] %q' "${method}" "${phase}" "${command}"
  printf ' %q' "$@"
  printf '\nLog: %s\n' "${log}"
  set +e
  "${command}" "$@" 2>&1 | tee "${log}"
  local rc="${PIPESTATUS[0]}"
  set -e
  if has_oom "${log}"; then
    printf '[%s/%s] detected OOM; stopping this method phase\n' "${method}" "${phase}" >&2
    record "${method}" "${phase}" oom "${OOM_RC}" "${log}"
    return "${OOM_RC}"
  fi
  if [[ "${rc}" -ne 0 ]]; then
    record "${method}" "${phase}" failed "${rc}" "${log}"
    return "${rc}"
  fi
  record "${method}" "${phase}" ok 0 "${log}"
  return 0
}

prepare_full() {
  local log="${STATUS_ROOT}/logs/prepare_$(date +%Y%m%d-%H%M%S).log"
  printf '[data/prepare] full native+low preparation\nLog: %s\n' "${log}"
  set +e
  "${SCRIPT_DIR}/prepare_full_native.sh" 2>&1 | tee "${log}"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    printf 'Full data preparation failed with rc=%s\n' "${rc}" >&2
    exit "${rc}"
  fi
}

if [[ "${prepare}" -eq 1 ]]; then
  prepare_full
fi

for manifest in "${LOW_MANIFEST}" "${NATIVE_MANIFEST}"; do
  require_file "${manifest}"
done
records="$(wc -l < "${NATIVE_MANIFEST}")"
if [[ "${records}" -lt 1 ]]; then
  printf 'Native counterfactual manifest is empty: %s\n' "${NATIVE_MANIFEST}" >&2
  exit 1
fi
if [[ "${prepare_only}" -eq 1 ]]; then
  printf 'Preparation complete; status: %s\n' "${STATUS_TSV}"
  exit 0
fi

nativeize_method() {
  local method="$1" checkpoint="$2"
  local low_canonical="${OUTPUT_ROOT}/${method}/full/low_canonical"
  local native_canonical="${OUTPUT_ROOT}/${method}/full/canonical"
  run_phase "${method}" nativeize env \
    "PYTHONPATH=${PROJECT_ROOT}" \
    "${PREP_PYTHON}" -m m2h_baselines.nativeize \
    --low-canonical "${low_canonical}" \
    --native-manifest "${NATIVE_MANIFEST}" \
    --output-dir "${native_canonical}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --low-content-box "${M2H_LOW_CONTENT_BOX:-64,0,448,512}"
}

verify_native_method() {
  local method="$1" checkpoint="$2"
  local native_canonical="${OUTPUT_ROOT}/${method}/full/canonical"
  run_phase "${method}" verify env \
    "PYTHONPATH=${PROJECT_ROOT}" \
    "${PREP_PYTHON}" "${PROJECT_ROOT}/scripts/verify_inference_outputs.py" \
    --canonical "${native_canonical}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --records "${records}" \
    --width 768 --height 1024 \
    --frozen-protocol "${DATA_ROOT}/eval/cf_subset.json" \
    --protocol-sha256 "${PROTOCOL_SHA256}"
}

run_method() {
  local method="$1"
  local train_script="${SCRIPT_DIR}/train_${method}.sh"
  local infer_script="${SCRIPT_DIR}/infer_${method}.sh"
  local low_canonical="${OUTPUT_ROOT}/${method}/full/low_canonical"
  local native_canonical="${OUTPUT_ROOT}/${method}/full/canonical"
  local checkpoint=""
  require_file "${train_script}"
  require_file "${infer_script}"

  if ! run_phase "${method}" train "${train_script}" full; then
    printf '[%s] training did not complete; continuing with next method\n' "${method}" >&2
    return 0
  fi

  if ! run_phase "${method}" inference env \
      "M2H_CANONICAL_DIR=${low_canonical}" \
      "${infer_script}" full; then
    printf '[%s] inference did not complete; continuing with next method\n' "${method}" >&2
    return 0
  fi

  if ! checkpoint="$("${PREP_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "${low_canonical}/summary.json")"; then
    printf '[%s] cannot read low canonical checkpoint summary; continuing with next method\n' "${method}" >&2
    record "${method}" checkpoint failed 1 "${low_canonical}/summary.json"
    return 0
  fi
  if ! nativeize_method "${method}" "${checkpoint}"; then
    printf '[%s] nativeization failed; continuing with next method\n' "${method}" >&2
    return 0
  fi
  if ! verify_native_method "${method}" "${checkpoint}"; then
    printf '[%s] native output verification failed; continuing with next method\n' "${method}" >&2
    return 0
  fi
  if ! run_phase "${method}" metrics env \
      "M2H_CANONICAL_DIR=${native_canonical}" \
      "M2H_EVAL_RESOLUTION=native" \
      "M2H_METRICS_MANIFEST=${NATIVE_MANIFEST}" \
      "${SCRIPT_DIR}/evaluate_metrics_v2.sh" "${method}" full; then
    printf '[%s] metrics failed; continuing with next method\n' "${method}" >&2
    return 0
  fi
  printf '[%s] full native pipeline completed\n' "${method}"
}

for method in refton ita_mdt idm_vton mcld ominicontrol; do
  run_method "${method}"
done

printf '\nFull queue finished. Per-phase status: %s\n' "${STATUS_TSV}"
printf 'Native outputs: %s/<method>/full/canonical\n' "${OUTPUT_ROOT}"
printf 'Reports: %s/generated/<method>/full\n' "${REPORT_ROOT}"
