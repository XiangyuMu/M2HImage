#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/OminiControl"
python="$(method_python ominicontrol)"
config="${M2H_OMINI_CONFIG:-${PROJECT_ROOT}/configs/methods/ominicontrol_${profile}.yaml}"
manifest="${PREPARED_ROOT}/low/counterfactual/samples.jsonl"
raw_dir="${OUTPUT_ROOT}/ominicontrol/${profile}/raw"
if [[ -n "${M2H_CHECKPOINT:-}" ]]; then
  checkpoint="${M2H_CHECKPOINT}"
else
  checkpoint="$(latest_dir_containing "${RUN_ROOT}/ominicontrol/${profile}" mannequin.safetensors)"
fi
require_file "${python}"
require_file "${config}"
require_file "${manifest}"
require_file "${checkpoint}/mannequin.safetensors"
require_file "${checkpoint}/identity.safetensors"
require_file "${checkpoint}/pose.safetensors"
require_file "${checkpoint}/m2h_checkpoint.json"
env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.verify_checkpoint \
  --method ominicontrol \
  --checkpoint "${checkpoint}" \
  --protocol-sha256 "${PROTOCOL_SHA256}"
mkdir -p "${raw_dir}"

limit_args=()
if [[ -n "${M2H_INFER_LIMIT:-}" ]]; then
  limit_args=(--limit "${M2H_INFER_LIMIT}")
fi

pushd "${repo}" >/dev/null
run_logged ominicontrol "infer_${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_INFER_GPU:-0}" \
  "${python}" -m omini.train_flux.infer_m2h \
  --config "${config}" \
  --manifest "${manifest}" \
  --prepared-root "${PREPARED_ROOT}" \
  --lora-dir "${checkpoint}" \
  --output-dir "${raw_dir}" \
  --seed 42 \
  --device cuda:0 \
  --memory-mode "${M2H_INFER_MEMORY_MODE:-group_offload}" \
  --group-offload-blocks "${M2H_INFER_GROUP_BLOCKS:-2}" \
  "${limit_args[@]}"
popd >/dev/null

collect_method_outputs ominicontrol "${profile}" "${checkpoint}" "${raw_dir}"
printf 'OminiControl canonical outputs: %s\n' "${OUTPUT_ROOT}/ominicontrol/${profile}/canonical"
