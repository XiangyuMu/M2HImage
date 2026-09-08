#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/OminiControl"
python="$(method_python ominicontrol)"
accelerate="$(method_accelerate ominicontrol)"
config="${M2H_OMINI_CONFIG:-${PROJECT_ROOT}/configs/methods/ominicontrol_${profile}.yaml}"
prompt_cache="${M2H_OMINI_PROMPT_CACHE:-${CACHE_ROOT}/models/derived/ominicontrol/m2h_prompt_256.safetensors}"
manifest="${PREPARED_ROOT}/low/train/samples.jsonl"
steps="${M2H_STEPS:-60000}"
checkpoint_steps="${M2H_CHECKPOINT_STEPS:-1000}"
if [[ "${profile}" == "smoke" ]]; then
  steps="${M2H_STEPS:-100}"
  checkpoint_steps="${M2H_CHECKPOINT_STEPS:-100}"
fi
accumulation="${M2H_GRAD_ACCUM:-8}"
save_batches="$((checkpoint_steps * accumulation))"
require_file "${python}"
require_file "${accelerate}"
require_file "${config}"
require_file "${manifest}"
require_file "${PROJECT_ROOT}/scripts/prepare_omini_prompt_cache.py"
require_dir "${MODEL_ROOT}/ominicontrol/FLUX.1-dev/transformer"

env "CUDA_VISIBLE_DEVICES=${M2H_OMINI_PROMPT_GPU:-0}" \
  "${python}" "${PROJECT_ROOT}/scripts/prepare_omini_prompt_cache.py" \
  --flux-path "${MODEL_ROOT}/ominicontrol/FLUX.1-dev" \
  --output "${prompt_cache}" \
  --max-sequence-length 256

pushd "${repo}" >/dev/null
run_logged ominicontrol "${profile}" env \
  "OMINI_CONFIG=${config}" \
  "M2H_OMINI_PROMPT_CACHE=${prompt_cache}" \
  "M2H_USE_FSDP=${M2H_USE_FSDP:-1}" \
  "M2H_MAX_STEPS=${steps}" \
  "M2H_SAVE_INTERVAL_BATCHES=${save_batches}" \
  "M2H_SAVE_PATH=${RUN_ROOT}/ominicontrol/${profile}" \
  "M2H_TASK=mannequin_to_human" \
  "M2H_DISABLE_TRAIN_SAMPLES=${M2H_DISABLE_TRAIN_SAMPLES:-1}" \
  "PATH=$(dirname "${python}"):${PATH}" \
  "${accelerate}" launch \
  --num_processes "${M2H_NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${M2H_PORT:-29535}" \
  -m omini.train_flux.train_m2h
popd >/dev/null
