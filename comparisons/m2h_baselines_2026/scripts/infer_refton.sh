#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/RefTon"
python="$(method_python refton)"
data_dir="${LAYOUT_ROOT}/low/counterfactual/refton"
base_model="${MODEL_ROOT}/refton/FLUX.1-Kontext-dev"
checkpoint="${M2H_CHECKPOINT:-${RUN_ROOT}/refton/${profile}}"
raw_dir="${OUTPUT_ROOT}/refton/${profile}/raw"
require_file "${python}"
require_dir "${data_dir}/test"
require_dir "${base_model}/transformer"
require_file "${checkpoint}/pytorch_lora_weights.safetensors"
if [[ "${profile}" == "full" || -f "${checkpoint}/m2h_checkpoint.json" ]]; then
  env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.verify_checkpoint \
    --method refton \
    --checkpoint "${checkpoint}" \
    --protocol-sha256 "${PROTOCOL_SHA256}"
fi
mkdir -p "${raw_dir}"

pushd "${repo}" >/dev/null
run_logged refton "infer_${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_INFER_GPU:-0}" \
  "${python}" inference.py \
  --pretrained_model_name_or_path "${base_model}" \
  --instance_data_dir "${data_dir}" \
  --dataset_type m2h \
  --split test \
  --output_dir "${checkpoint}" \
  --inference_output_dir "${raw_dir}" \
  --inference_memory_mode "${M2H_INFER_MEMORY_MODE:-group_offload}" \
  --inference_group_offload_blocks "${M2H_INFER_GROUP_BLOCKS:-2}" \
  --instance_prompt "Transform the mannequin into the referenced person while preserving the outfit, pose, and background." \
  --use_person \
  --use_reference \
  --mixed_precision bf16 \
  --height 512 \
  --width 512 \
  --inference_batch_size "${M2H_INFER_BATCH:-1}" \
  --dataloader_num_workers "${M2H_DATALOADER_WORKERS:-2}" \
  --cond_scale 1.0 \
  --seed 42 \
  --report_to none \
  --max_sequence_length 256
popd >/dev/null

collect_method_outputs refton "${profile}" "${checkpoint}" "${raw_dir}"
printf 'RefTon canonical outputs: %s\n' "${OUTPUT_ROOT}/refton/${profile}/canonical"
