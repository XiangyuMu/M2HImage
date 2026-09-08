#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

python="$(method_python idm_vton)"
converter="${PROJECT_ROOT}/scripts/convert_idm_unet_bf16.py"
source_unet="${MODEL_ROOT}/idm_vton/IDM-VTON/unet"
derived_root="${CACHE_ROOT}/models/derived/idm_vton/IDM-VTON-bf16"
derived_unet="${derived_root}/unet"
destination="${MODEL_ROOT}/idm_vton/IDM-VTON-bf16"
marker="${derived_unet}/diffusion_pytorch_model.safetensors"

require_file "${python}"
require_file "${converter}"
require_file "${source_unet}/config.json"
require_file "${source_unet}/diffusion_pytorch_model.bin"

if [[ ! -f "${marker}" ]]; then
  "${python}" "${converter}" \
    --source-unet "${source_unet}" \
    --output-unet "${derived_unet}" \
    --max-shard-size-gb "${M2H_IDM_BF16_SHARD_GB:-10}"
fi
require_file "${marker}"

mkdir -p "$(dirname "${destination}")"
if [[ -L "${destination}" ]]; then
  if [[ "$(readlink -f "${destination}")" != "$(readlink -f "${derived_root}")" ]]; then
    printf 'Refusing to replace derived-model symlink: %s\n' "${destination}" >&2
    exit 1
  fi
elif [[ -e "${destination}" ]]; then
  printf 'Refusing to replace derived-model path: %s\n' "${destination}" >&2
  exit 1
else
  ln -s "$(readlink -f "${derived_root}")" "${destination}"
fi

printf 'IDM-VTON BF16 UNet ready: %s -> %s\n' \
  "${destination}" "$(readlink -f "${destination}")"
