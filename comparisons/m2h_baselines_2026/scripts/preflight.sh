#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

requested="${1:-all}"
kind="${2:-paired}"
profile="${3:-smoke}"
case "${kind}" in
  paired|counterfactual) ;;
  *) printf 'Kind must be paired or counterfactual: %s\n' "${kind}" >&2; exit 2 ;;
esac
require_profile "${profile}"

require_file "${DATA_ROOT}/eval/cf_subset.json"
actual_protocol_sha="$(sha256sum "${DATA_ROOT}/eval/cf_subset.json" | awk '{print $1}')"
if [[ "${actual_protocol_sha}" != "${PROTOCOL_SHA256}" ]]; then
  printf 'Frozen protocol hash mismatch: got %s, expected %s\n' \
    "${actual_protocol_sha}" "${PROTOCOL_SHA256}" >&2
  exit 1
fi

require_command nvidia-smi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( gpu_count < M2H_NUM_PROCESSES )); then
  printf 'Need %s GPUs but only %s are visible.\n' "${M2H_NUM_PROCESSES}" "${gpu_count}" >&2
  exit 1
fi
printf 'Visible GPUs: %s\n' "${gpu_count}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
df -h / "${ENV_ROOT}" "${CACHE_ROOT}" 2>/dev/null || true

probe_one() {
  local method="$1"
  local python
  require_method "${method}"
  python="$(method_python "${method}")"
  require_file "${python}"
  require_file "${PREPARED_ROOT}/low/$([[ "${kind}" == "paired" ]] && printf train || printf counterfactual)/samples.jsonl"

  case "${method}" in
    refton)
      require_dir "${LAYOUT_ROOT}/low/${kind}/refton/$([[ "${kind}" == "paired" ]] && printf train || printf test)"
      require_dir "${MODEL_ROOT}/refton/FLUX.1-Kontext-dev/transformer"
      "${python}" -c 'import accelerate, diffusers, torch, transformers'
      ;;
    ita_mdt)
      require_dir "${LAYOUT_ROOT}/low/${kind}/ita_mdt/zalando-hd-resized/$([[ "${kind}" == "paired" ]] && printf train || printf test)"
      require_dir "${MODEL_ROOT}/ita_mdt/sdxl-inpainting/vae"
      require_dir "${MODEL_ROOT}/ita_mdt/dinov2"
      require_file "${TORCH_HOME}/hub/checkpoints/dinov2_vitg14_pretrain.pth"
      "${python}" -c 'import albumentations, cv2, diffusers, torch'
      ;;
    idm_vton)
      require_dir "${LAYOUT_ROOT}/low/${kind}/idm_vton/$([[ "${kind}" == "paired" ]] && printf train || printf test)"
      require_dir "${MODEL_ROOT}/idm_vton/IDM-VTON/unet"
      require_file "${MODEL_ROOT}/idm_vton/ip-adapter-plus_sdxl_vit-h.bin"
      "${python}" -c 'import accelerate, diffusers, torch, transformers'
      ;;
    mcld)
      require_file "${LAYOUT_ROOT}/low/${kind}/mcld/$([[ "${kind}" == "paired" ]] && printf train || printf test).csv"
      require_dir "${MODEL_ROOT}/mcld/sd-image-variations-diffusers/unet"
      require_file "${MODEL_ROOT}/mcld/control_v11p_sd15_seg/diffusion_pytorch_model.bin"
      "${python}" -c 'import insightface, mlflow, omegaconf, torch'
      ;;
    ominicontrol)
      require_dir "${MODEL_ROOT}/ominicontrol/FLUX.1-dev/transformer"
      "${python}" -c 'import diffusers, lightning, torch'
      ;;
  esac

  env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.preflight \
    --method "${method}" \
    --project-root "${PROJECT_ROOT}" \
    --layout-root "${LAYOUT_ROOT}" \
    --prepared-root "${PREPARED_ROOT}" \
    --kind "${kind}" \
    --profile "${profile}"
}

if [[ "${requested}" == "all" ]]; then
  for method in refton ita_mdt idm_vton mcld ominicontrol; do
    probe_one "${method}"
  done
else
  probe_one "${requested}"
fi

printf 'Preflight passed: method=%s kind=%s profile=%s protocol=%s\n' \
  "${requested}" "${kind}" "${profile}" "${actual_protocol_sha}"
