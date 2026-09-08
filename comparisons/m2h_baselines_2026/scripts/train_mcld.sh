#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/MCLD"
python="$(method_python mcld)"
accelerate="$(method_accelerate mcld)"
config="${M2H_MCLD_CONFIG:-${PROJECT_ROOT}/configs/methods/mcld_${profile}.yaml}"
data_dir="${LAYOUT_ROOT}/low/paired/mcld"
steps="${M2H_STEPS:-60000}"
model_save_epoch_interval="${M2H_MODEL_SAVE_EPOCH_INTERVAL:-1}"
if [[ "${profile}" == "smoke" ]]; then
  steps="${M2H_STEPS:-100}"
  # Component checkpoints total about 7 GiB. The 32-sample smoke split has
  # only two optimizer updates per epoch, so saving every epoch would rewrite
  # hundreds of GiB without adding a validation point. train.py always saves
  # once more when max_train_steps is reached.
  model_save_epoch_interval="${M2H_MODEL_SAVE_EPOCH_INTERVAL:-1000000}"
fi
require_file "${python}"
require_file "${accelerate}"
require_file "${config}"
require_file "${data_dir}/train.csv"
require_file "${data_dir}/test.csv"
require_dir "${data_dir}/train_face"
require_dir "${MODEL_ROOT}/mcld/sd-image-variations-diffusers"
require_dir "${MODEL_ROOT}/mcld/stable-diffusion-v1-5"
require_file "${MODEL_ROOT}/mcld/control_v11p_sd15_seg/diffusion_pytorch_model.bin"
site_packages="$("${python}" -c 'import site; print(site.getsitepackages()[0])')"
cusparse_dir="${site_packages}/nvidia/cusparse/lib"
torch_lib_dir="${site_packages}/torch/lib"
if [[ ! -f "${cusparse_dir}/libcusparse.so.11" ]]; then
  cusparse_dir="${M2H_CUDA11_CUSPARSE_DIR:-/home/muxiangyu/miniconda3/envs/gsinpaint/lib/python3.10/site-packages/nvidia/cusparse/lib}"
fi
require_file "${cusparse_dir}/libcusparse.so.11"
require_dir "${torch_lib_dir}"
mkdir -p "${RUN_ROOT}/mcld/mlruns"

validation_args=(--skip_validation)
if [[ "${M2H_ENABLE_INLINE_VALIDATION:-0}" == "1" ]]; then
  validation_args=()
fi

launch_args=(
  --num_processes "${M2H_NUM_PROCESSES}"
  --mixed_precision fp16
  --main_process_port "${M2H_PORT:-29534}"
)
memory_args=()
if [[ "${M2H_USE_FSDP:-1}" == "1" ]]; then
  launch_args+=(
    --use_fsdp
    --fsdp_sharding_strategy 1
    --fsdp_auto_wrap_policy NO_WRAP
    --fsdp_backward_prefetch_policy BACKWARD_PRE
    --fsdp_state_dict_type FULL_STATE_DICT
    --fsdp_forward_prefetch false
    --fsdp_use_orig_params true
    --fsdp_sync_module_states false
  )
  memory_args+=(--disable_gradient_checkpointing)
fi

pushd "${repo}" >/dev/null
run_logged mcld "${profile}" env \
  "MLFLOW_TRACKING_URI=file://${RUN_ROOT}/mcld/mlruns" \
  "PATH=$(dirname "${python}"):${PATH}" \
  "LD_LIBRARY_PATH=${cusparse_dir}:${torch_lib_dir}:${LD_LIBRARY_PATH:-}" \
  "${accelerate}" launch \
  "${launch_args[@]}" \
  train.py \
  --config "${config}" \
  --output_dir "${RUN_ROOT}/mcld/${profile}" \
  --exp_name "mcld_m2h_${profile}" \
  --max_train_steps "${steps}" \
  --save_model_epoch_interval "${model_save_epoch_interval}" \
  "${validation_args[@]}" \
  "${memory_args[@]}"
popd >/dev/null
