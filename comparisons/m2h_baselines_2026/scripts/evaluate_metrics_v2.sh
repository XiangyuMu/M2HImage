#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

method="${1:?usage: evaluate_metrics_v2.sh METHOD [smoke|full]}"
profile="${2:-smoke}"
require_method "${method}"
require_profile "${profile}"

canonical="${M2H_CANONICAL_DIR:-${OUTPUT_ROOT}/${method}/${profile}/canonical}"
eval_resolution="${M2H_EVAL_RESOLUTION:-low}"
manifest="${M2H_METRICS_MANIFEST:-${PREPARED_ROOT}/${eval_resolution}/counterfactual/samples.jsonl}"
main_config="${M2H_METRICS_CONFIG:-${PROJECT_ROOT}/configs/metrics_v2.yaml}"
metrics_python="${M2H_METRICS_PYTHON:-${ENV_ROOT}/mcld/bin/python}"
prep_python="${M2H_PREP_PYTHON:-$(method_python refton)}"
run_name="${method}_${profile}"
report_dir="${REPORT_ROOT}/generated/${method}/${profile}"
generated_config="${report_dir}/metrics_config.yaml"
require_file "${canonical}/summary.json"
require_file "${manifest}"
require_file "${main_config}"
require_file "${METRICS_MAIN_ROOT}/eval_metrics_v2.py"
require_file "${metrics_python}"
require_file "${prep_python}"
mkdir -p "${report_dir}"
run_dir="${report_dir}/${run_name}"
if [[ -e "${run_dir}" ]]; then
  archive_dir="${report_dir}/archive/${run_name}_$(date +%Y%m%d-%H%M%S)_pid$$"
  mkdir -p "$(dirname "${archive_dir}")"
  mv "${run_dir}" "${archive_dir}"
  printf 'Archived stale metrics cache to %s\n' "${archive_dir}"
fi

env "PYTHONPATH=${PROJECT_ROOT}" "${prep_python}" -m m2h_baselines.metrics_config \
  --base-config "${main_config}" \
  --manifest "${manifest}" \
  --gen-dir "${canonical}" \
  --data-root "${DATA_ROOT}" \
  --output-config "${generated_config}" \
  --output-root "${report_dir}" \
  --run-name "${run_name}" \
  --label "${method} ${profile}"

smoke_args=()
if [[ "${profile}" == "smoke" || -n "${M2H_INFER_LIMIT:-}" ]]; then
  smoke_args=(--smoke)
fi

pushd "${METRICS_MAIN_ROOT}" >/dev/null
run_logged "metrics_${method}" "${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_METRICS_GPUS:-0,1}" \
  "${metrics_python}" eval_metrics_v2.py \
  --config "${generated_config}" \
  --subset "${generated_config%.yaml}.protocol.json" \
  --run "${run_name}" \
  --metrics "${M2H_METRICS:-all}" \
  --device "${M2H_METRICS_DEVICE:-cuda:0}" \
  --pose-device "${M2H_POSE_DEVICE:-cuda:1}" \
  --output-root "${report_dir}" \
  "${smoke_args[@]}"
popd >/dev/null

verify_args=()
if [[ "${M2H_METRICS_ALLOW_QUALITY_FAILURES:-0}" == "1" ]]; then
  verify_args=(--allow-quality-failures)
fi

env "PYTHONPATH=${PROJECT_ROOT}" "${prep_python}" -m m2h_baselines.verify_metrics \
  --run-dir "${report_dir}/${run_name}" \
  --canonical-dir "${canonical}" \
  --profile "${profile}" \
  "${verify_args[@]}"

printf 'Metrics-v2 report directory: %s/%s\n' "${report_dir}" "${run_name}"
