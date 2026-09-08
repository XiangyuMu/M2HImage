#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/MCLD"
python="$(method_python mcld)"
config="${M2H_MCLD_CONFIG:-${PROJECT_ROOT}/configs/methods/mcld_${profile}.yaml}"
data_dir="${LAYOUT_ROOT}/low/counterfactual/mcld"
checkpoint="${M2H_CHECKPOINT:-${RUN_ROOT}/mcld/${profile}/mcld_m2h_${profile}}"
raw_dir="${OUTPUT_ROOT}/mcld/${profile}/raw"
require_file "${python}"
require_file "${config}"
require_file "${data_dir}/test.csv"
require_dir "${data_dir}/test_face"
require_dir "${checkpoint}"
require_file "${checkpoint}/m2h_checkpoint.json"
env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.verify_checkpoint \
  --method mcld \
  --checkpoint "${checkpoint}" \
  --protocol-sha256 "${PROTOCOL_SHA256}"
mkdir -p "${raw_dir}"

limit_args=()
if [[ -n "${M2H_INFER_LIMIT:-}" ]]; then
  limit_args=(--limit "${M2H_INFER_LIMIT}")
fi

pushd "${repo}" >/dev/null
run_logged mcld "infer_${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_INFER_GPU:-0}" \
  "${python}" test.py \
  --save_path "${raw_dir}" \
  --ckpt_dir "${checkpoint}" \
  --config_path "${config}" \
  --data_root "${data_dir}/" \
  --exp_name generated \
  --batch_size "${M2H_INFER_BATCH:-1}" \
  --num_workers "${M2H_DATALOADER_WORKERS:-2}" \
  "${limit_args[@]}"
popd >/dev/null

collect_method_outputs mcld "${profile}" "${checkpoint}" "${raw_dir}/generated/512"
printf 'MCLD canonical outputs: %s\n' "${OUTPUT_ROOT}/mcld/${profile}/canonical"
