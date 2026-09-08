#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/IDM-VTON"
python="$(method_python idm_vton)"
data_dir="${LAYOUT_ROOT}/low/counterfactual/idm_vton"
base_model="${MODEL_ROOT}/idm_vton/IDM-VTON"
raw_dir="${OUTPUT_ROOT}/idm_vton/${profile}/raw"
if [[ -n "${M2H_CHECKPOINT:-}" ]]; then
  checkpoint="${M2H_CHECKPOINT}"
else
  checkpoint_metadata="$(latest_file "${RUN_ROOT}/idm_vton/${profile}" m2h_checkpoint.json)"
  checkpoint="$(dirname "${checkpoint_metadata}")"
fi
require_file "${python}"
require_dir "${base_model}"
require_dir "${checkpoint}/unet"
require_file "${checkpoint}/m2h_checkpoint.json"
require_dir "${data_dir}/test"
env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.verify_checkpoint \
  --method idm_vton \
  --checkpoint "${checkpoint}" \
  --protocol-sha256 "${PROTOCOL_SHA256}"
mkdir -p "${raw_dir}"

memory_args=()
if "${python}" -c 'import xformers' >/dev/null 2>&1; then
  memory_args+=(--enable_xformers_memory_efficient_attention)
fi
limit_args=()
if [[ -n "${M2H_INFER_LIMIT:-}" ]]; then
  limit_args=(--limit "${M2H_INFER_LIMIT}")
fi

pushd "${repo}" >/dev/null
run_logged idm_vton "infer_${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_INFER_GPU:-0}" \
  "${python}" inference.py \
  --pretrained_model_name_or_path "${base_model}" \
  --pretrained_unet_path "${checkpoint}" \
  --data_dir "${data_dir}" \
  --dataset_type m2h \
  --output_dir "${raw_dir}" \
  --width 512 \
  --height 512 \
  --num_inference_steps "${M2H_INFER_STEPS:-30}" \
  --seed 42 \
  --test_batch_size "${M2H_INFER_BATCH:-1}" \
  --test_num_workers "${M2H_DATALOADER_WORKERS:-2}" \
  --guidance_scale 2.0 \
  --mixed_precision fp16 \
  "${memory_args[@]}" \
  "${limit_args[@]}"
popd >/dev/null

collect_method_outputs idm_vton "${profile}" "${checkpoint}" "${raw_dir}"
printf 'IDM-VTON canonical outputs: %s\n' "${OUTPUT_ROOT}/idm_vton/${profile}/canonical"
