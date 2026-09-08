#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/IDM-VTON"
python="$(method_python idm_vton)"
accelerate="$(method_accelerate idm_vton)"
data_dir="${LAYOUT_ROOT}/low/paired/idm_vton"
base_model="${MODEL_ROOT}/idm_vton/IDM-VTON"
bf16_model="${MODEL_ROOT}/idm_vton/IDM-VTON-bf16"
ip_adapter="${MODEL_ROOT}/idm_vton/ip-adapter-plus_sdxl_vit-h.bin"
output_dir="${RUN_ROOT}/idm_vton/${profile}"
steps="${M2H_STEPS:-60000}"
checkpoint_steps="${M2H_CHECKPOINT_STEPS:-1000}"
if [[ "${profile}" == "smoke" ]]; then
  steps="${M2H_STEPS:-100}"
  checkpoint_steps="${M2H_CHECKPOINT_STEPS:-100}"
fi
require_file "${python}"
require_file "${accelerate}"
require_dir "${data_dir}/train"
require_dir "${data_dir}/test"
require_dir "${base_model}/unet"
require_file "${ip_adapter}"
site_packages="$("${python}" -c 'import site; print(site.getsitepackages()[0])')"
cusparse_dir="${site_packages}/nvidia/cusparse/lib"
torch_lib_dir="${site_packages}/torch/lib"
if [[ ! -d "${cusparse_dir}" ]]; then
  cusparse_dir="${M2H_CUDA11_CUSPARSE_DIR:-/home/muxiangyu/miniconda3/envs/gsinpaint/lib/python3.10/site-packages/nvidia/cusparse/lib}"
fi
require_dir "${cusparse_dir}"

use_fsdp="${M2H_USE_FSDP:-1}"
if [[ "${use_fsdp}" == "1" ]]; then
  mixed_precision="${M2H_MIXED_PRECISION:-fp16}"
  trainable_weight_dtype="${M2H_TRAINABLE_WEIGHT_DTYPE:-fp32}"
else
  mixed_precision="${M2H_MIXED_PRECISION:-fp16}"
  trainable_weight_dtype="${M2H_TRAINABLE_WEIGHT_DTYPE:-fp32}"
fi
# A bf16 trainable SDXL U-Net overflows on the RTX 3090 setup after a few
# optimizer updates. Keep the stable fp32 master weights as the default while
# allowing an explicit override for hardware/configurations that support bf16.
if [[ "${use_fsdp}" == "1" ]]; then
  unet_model="${M2H_IDM_TRAINABLE_MODEL:-${bf16_model}}"
  require_file "${unet_model}/unet/diffusion_pytorch_model.safetensors"
else
  unet_model="${base_model}"
fi

launch_args=(
  --num_processes "${M2H_NUM_PROCESSES}"
  --mixed_precision "${mixed_precision}"
  --main_process_port "${M2H_PORT:-29533}"
)
if [[ "${use_fsdp}" == "1" ]]; then
  launch_args+=(
    --use_fsdp
    --fsdp_sharding_strategy 1
    --fsdp_auto_wrap_policy SIZE_BASED_WRAP
    --fsdp_min_num_params "${M2H_FSDP_MIN_NUM_PARAMS:-10000000}"
    --fsdp_backward_prefetch_policy BACKWARD_PRE
    --fsdp_state_dict_type FULL_STATE_DICT
    --fsdp_forward_prefetch false
    --fsdp_use_orig_params true
    --fsdp_cpu_ram_efficient_loading false
    --fsdp_sync_module_states false
  )
fi
mkdir -p "${output_dir}"

memory_args=(--gradient_checkpointing --offload_vae_between_steps)
# Native AdamW allocates two fp32 state tensors for the full SDXL UNet and
# exceeds 24 GiB at its first update. The 8-bit state is the validated 3090
# default; set M2H_USE_8BIT_ADAM=0 only on a larger-memory setup.
if [[ "${M2H_USE_8BIT_ADAM:-1}" == "1" ]]; then
  memory_args+=(--use_8bit_adam)
fi
if "${python}" -c 'import xformers' >/dev/null 2>&1; then
  memory_args+=(--enable_xformers_memory_efficient_attention)
fi
validation_args=(--skip_validation)
if [[ "${M2H_ENABLE_INLINE_VALIDATION:-0}" == "1" ]]; then
  validation_args=()
fi

pushd "${repo}" >/dev/null
run_logged idm_vton "${profile}" env \
  "PATH=$(dirname "${python}"):${PATH}" \
  "LD_LIBRARY_PATH=${cusparse_dir}:${torch_lib_dir}:${LD_LIBRARY_PATH:-}" \
  "${accelerate}" launch \
  "${launch_args[@]}" \
  train_xl.py \
  --pretrained_model_name_or_path "${base_model}" \
  --pretrained_garmentnet_path "${base_model}" \
  --pretrained_unet_path "${unet_model}" \
  --pretrained_ip_adapter_path "${ip_adapter}" \
  --image_encoder_path "${base_model}/image_encoder" \
  --data_dir "${data_dir}" \
  --dataset_type m2h \
  --output_dir "${output_dir}" \
  --width 512 \
  --height 512 \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --gradient_accumulation_steps "${M2H_GRAD_ACCUM:-8}" \
  --max_train_steps "${steps}" \
  --checkpointing_epoch "${checkpoint_steps}" \
  --logging_steps 1000 \
  --learning_rate "${M2H_LR:-1e-5}" \
  --snr_gamma 5.0 \
  --noise_offset 0.05 \
  --mixed_precision "${mixed_precision}" \
  --trainable_weight_dtype "${trainable_weight_dtype}" \
  --seed 42 \
  --train_num_workers "${M2H_DATALOADER_WORKERS:-4}" \
  --test_num_workers 2 \
  "${memory_args[@]}" \
  "${validation_args[@]}"
popd >/dev/null
