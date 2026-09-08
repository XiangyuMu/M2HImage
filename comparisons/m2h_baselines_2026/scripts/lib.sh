#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${M2H_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${M2H_DATA_ROOT:-/home/muxiangyu/datasets/M2HImage/M2H_Final_v2}"
ENV_ROOT="${M2H_ENV_ROOT:-/data/muxiangyu/envs/m2h_baselines_2026}"
CACHE_ROOT="${M2H_CACHE_ROOT:-/data/muxiangyu/cache/m2h_baselines_2026}"
PREPARED_ROOT="${M2H_PREPARED_ROOT:-${PROJECT_ROOT}/prepared}"
LAYOUT_ROOT="${M2H_LAYOUT_ROOT:-${PROJECT_ROOT}/layouts}"
MODEL_ROOT="${M2H_MODEL_ROOT:-${PROJECT_ROOT}/models}"
RUN_ROOT="${M2H_RUN_ROOT:-${PROJECT_ROOT}/runs}"
OUTPUT_ROOT="${M2H_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
REPORT_ROOT="${M2H_REPORT_ROOT:-${PROJECT_ROOT}/reports}"
CONFIG_PATH="${M2H_CONFIG:-${PROJECT_ROOT}/configs/study.yaml}"
METRICS_MAIN_ROOT="${M2H_MAIN_REPO:-/data/muxiangyu/programs/M2HImage}"
PROTOCOL_SHA256="7fa287b79ffe32e6bbf8e8d20b0082a74e20c696997abf669f7b80d29a753985"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
M2H_NUM_PROCESSES="${M2H_NUM_PROCESSES:-2}"

export PROJECT_ROOT DATA_ROOT ENV_ROOT CACHE_ROOT PREPARED_ROOT LAYOUT_ROOT
export MODEL_ROOT RUN_ROOT OUTPUT_ROOT REPORT_ROOT CONFIG_PATH
export METRICS_MAIN_ROOT PROTOCOL_SHA256
export CUDA_VISIBLE_DEVICES M2H_NUM_PROCESSES

configure_runtime() {
  mkdir -p \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/tmp" \
    "${RUN_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${REPORT_ROOT}"
  export HF_HOME="${CACHE_ROOT}/huggingface"
  export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
  export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/hub"
  export TORCH_HOME="${CACHE_ROOT}/torch"
  export XDG_CACHE_HOME="${CACHE_ROOT}"
  export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
  export TMPDIR="${CACHE_ROOT}/tmp"
  export TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
}

method_python() {
  local method="$1"
  printf '%s\n' "${ENV_ROOT}/${method}/bin/python"
}

method_accelerate() {
  local method="$1"
  printf '%s\n' "${ENV_ROOT}/${method}/bin/accelerate"
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'Missing command: %s\n' "${command_name}" >&2
    return 1
  }
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    printf 'Missing file: %s\n' "${path}" >&2
    return 1
  }
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || {
    printf 'Missing directory: %s\n' "${path}" >&2
    return 1
  }
}

require_method() {
  case "$1" in
    refton|ita_mdt|idm_vton|mcld|ominicontrol) ;;
    *)
      printf 'Unknown method: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

require_profile() {
  case "$1" in
    smoke|full) ;;
    *) printf 'Profile must be smoke or full, got: %s\n' "$1" >&2; return 2 ;;
  esac
}

profile_from_invocation() {
  local explicit="${1:-}"
  case "$(basename "$0")" in
    *_smoke.sh) printf '%s\n' smoke ;;
    *_full.sh) printf '%s\n' full ;;
    *)
      if [[ -n "${explicit}" ]]; then
        printf '%s\n' "${explicit}"
      else
        printf '%s\n' smoke
      fi
      ;;
  esac
}

run_logged() {
  local method="$1"
  local profile="$2"
  shift 2
  local log_dir="${RUN_ROOT}/${method}/${profile}/logs"
  local log_path="${log_dir}/$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "${log_dir}"
  printf 'Log: %s\n' "${log_path}"
  "$@" 2>&1 | tee "${log_path}"
}

latest_file() {
  local root="$1"
  local pattern="$2"
  local result
  result="$(find "${root}" -type f -name "${pattern}" -print 2>/dev/null | sort -V | tail -n 1)"
  [[ -n "${result}" ]] || {
    printf 'No file matching %s under %s\n' "${pattern}" "${root}" >&2
    return 1
  }
  printf '%s\n' "${result}"
}

latest_dir_containing() {
  local root="$1"
  local filename="$2"
  local result
  result="$(find "${root}" -type f -name "${filename}" -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -n 1)"
  [[ -n "${result}" ]] || {
    printf 'No directory containing %s under %s\n' "${filename}" "${root}" >&2
    return 1
  }
  printf '%s\n' "${result#* }"
}

collect_method_outputs() {
  local method="$1"
  local profile="$2"
  local checkpoint="$3"
  local raw_dir="$4"
  local prep_python="${M2H_PREP_PYTHON:-$(method_python refton)}"
  if [[ ! -x "${prep_python}" ]]; then
    prep_python="/home/muxiangyu/miniconda3/bin/python"
  fi
  local limit_args=()
  if [[ -n "${M2H_INFER_LIMIT:-}" ]]; then
    limit_args=(--limit "${M2H_INFER_LIMIT}")
  fi
  local canonical_dir="${M2H_CANONICAL_DIR:-${OUTPUT_ROOT}/${method}/${profile}/canonical}"
  env "PYTHONPATH=${PROJECT_ROOT}" "${prep_python}" -m m2h_baselines.collect \
    --manifest "${PREPARED_ROOT}/low/counterfactual/samples.jsonl" \
    --raw-dir "${raw_dir}" \
    --output-dir "${canonical_dir}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    "${limit_args[@]}"
}

configure_runtime
