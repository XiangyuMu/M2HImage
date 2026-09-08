#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/RefTon"
python="$(method_python refton)"
accelerate="$(method_accelerate refton)"
data_dir="${LAYOUT_ROOT}/low/paired/refton"
model_dir="${MODEL_ROOT}/refton/FLUX.1-Kontext-dev"
output_dir="${RUN_ROOT}/refton/${profile}"
require_file "${python}"
require_file "${accelerate}"
require_dir "${data_dir}/train"

launch_args=(
  --num_processes "${M2H_NUM_PROCESSES}"
  --mixed_precision bf16
  --main_process_port "${M2H_PORT:-29531}"
)
if [[ "${M2H_USE_FSDP:-1}" == "1" ]]; then
  launch_args+=(
    --use_fsdp
    --fsdp_version 1
    --fsdp_sharding_strategy FULL_SHARD
    --fsdp_auto_wrap_policy SIZE_BASED_WRAP
    # The 1M threshold keeps individual FLUX projection all-gathers small
    # enough to coexist with accumulated rank-64 LoRA gradients on 24 GiB.
    --fsdp_min_num_params "${M2H_FSDP_MIN_NUM_PARAMS:-1000000}"
    --fsdp_backward_prefetch BACKWARD_PRE
    --fsdp_state_dict_type FULL_STATE_DICT
    --fsdp_forward_prefetch false
    --fsdp_use_orig_params true
    --fsdp_cpu_ram_efficient_loading false
    --fsdp_sync_module_states false
  )
fi
require_dir "${model_dir}/transformer"
mkdir -p "${output_dir}"

duration=(--num_train_epochs "${M2H_EPOCHS:-64}")
if [[ "${profile}" == "smoke" || -n "${M2H_STEPS:-}" ]]; then
  duration=(--max_train_steps "${M2H_STEPS:-100}")
fi

pushd "${repo}" >/dev/null
run_logged refton "${profile}" env \
  "PATH=$(dirname "${python}"):${PATH}" \
  "${accelerate}" launch \
  "${launch_args[@]}" \
  train_refton_lora.py \
  --pretrained_model_name_or_path "${model_dir}" \
  --instance_data_dir "${data_dir}" \
  --dataset_type m2h \
  --split train \
  --output_dir "${output_dir}" \
  --cache_dir "${CACHE_ROOT}/huggingface" \
  --instance_prompt "Transform the mannequin into the referenced person while preserving the outfit, pose, and background." \
  --mixed_precision bf16 \
  --height 512 \
  --width 512 \
  --center_crop \
  --train_batch_size 1 \
  --gradient_accumulation_steps "${M2H_REFTON_GRAD_ACCUM:-${M2H_GRAD_ACCUM:-8}}" \
  --guidance_scale 1 \
  --gradient_checkpointing \
  --optimizer adamw \
  --rank "${M2H_LORA_RANK:-64}" \
  --lora_alpha "${M2H_LORA_ALPHA:-128}" \
  --use_8bit_adam \
  --learning_rate "${M2H_LR:-1e-4}" \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --cond_scale 1.0 \
  --seed 42 \
  --dropout_reference 0.5 \
  --person_prob 0.5 \
  --max_sequence_length 256 \
  --dataloader_num_workers "${M2H_DATALOADER_WORKERS:-4}" \
  --checkpointing_steps "${M2H_CHECKPOINT_STEPS:-1000}" \
  --report_to none \
  --allow_tf32 \
  "${duration[@]}"
popd >/dev/null

env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.record_checkpoint \
  --method refton \
  --run-dir "${output_dir}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --base-model "${model_dir}" \
  --data-dir "${data_dir}"
