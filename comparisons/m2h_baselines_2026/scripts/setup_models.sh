#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

requested="${1:-all}"
MODEL_STORE="${M2H_MODEL_STORE:-${CACHE_ROOT}/models}"
mkdir -p "${MODEL_ROOT}" "${MODEL_STORE}" "${CACHE_ROOT}/downloads"

wants() {
  [[ "${requested}" == "all" || "${requested}" == "$1" ]]
}

link_target() {
  local source="$1"
  local destination="$2"
  require_file_or_dir "${source}"
  mkdir -p "$(dirname "${destination}")"
  if [[ -L "${destination}" ]]; then
    if [[ "$(readlink -f "${destination}")" == "$(readlink -f "${source}")" ]]; then
      return
    fi
    printf 'Refusing to replace model symlink: %s\n' "${destination}" >&2
    return 1
  fi
  if [[ -e "${destination}" ]]; then
    printf 'Model target already exists and is not a symlink: %s\n' "${destination}" >&2
    return 1
  fi
  ln -s "$(readlink -f "${source}")" "${destination}"
}

require_file_or_dir() {
  [[ -e "$1" ]] || {
    printf 'Missing model source: %s\n' "$1" >&2
    return 1
  }
}

download_exact_file() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"
  local expected_sha256="${4:-}"
  local actual_bytes=0
  local actual_sha256

  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]]; then
    actual_bytes="$(stat -Lc '%s' "${destination}")"
  fi
  if (( actual_bytes > expected_bytes )); then
    printf 'Downloaded file is larger than expected (%s > %s): %s\n' \
      "${actual_bytes}" "${expected_bytes}" "${destination}" >&2
    return 1
  fi
  if (( actual_bytes < expected_bytes )); then
    printf 'Downloading/resuming %s at byte %s of %s.\n' \
      "${destination}" "${actual_bytes}" "${expected_bytes}"
    local attempt aria2_succeeded=0
    if command -v aria2c >/dev/null 2>&1 && [[ "${M2H_DISABLE_ARIA2:-0}" != "1" ]]; then
      if aria2c \
        --continue=true \
        --max-connection-per-server=8 \
        --split=8 \
        --min-split-size=1M \
        --file-allocation=none \
        --auto-file-renaming=false \
        --max-tries=20 \
        --retry-wait=5 \
        --connect-timeout=30 \
        --timeout=60 \
        --summary-interval=30 \
        --dir="$(dirname "${destination}")" \
        --out="$(basename "${destination}")" \
        "${url}"; then
        aria2_succeeded=1
      else
        if [[ -f "${destination}.aria2" ]]; then
          printf 'aria2c failed with an active control file; refusing an unsafe curl resume: %s\n' \
            "${destination}.aria2" >&2
          return 1
        fi
        printf 'aria2c failed without a control file; falling back to resumable curl attempts.\n' >&2
      fi
    fi
    if (( aria2_succeeded == 0 )); then
      for ((attempt = 1; attempt <= 20; attempt++)); do
        if curl -fL --retry 5 --continue-at - -o "${destination}" "${url}"; then
          break
        fi
        if (( attempt == 20 )); then
          printf 'Download failed after %s resumable attempts: %s\n' \
            "${attempt}" "${destination}" >&2
          return 1
        fi
        printf 'Download attempt %s failed; resuming again in 5 seconds.\n' \
          "${attempt}" >&2
        sleep 5
      done
    fi
  fi
  actual_bytes="$(stat -Lc '%s' "${destination}")"
  [[ "${actual_bytes}" -eq "${expected_bytes}" ]] || {
    printf 'Downloaded file has the wrong size (%s != %s): %s\n' \
      "${actual_bytes}" "${expected_bytes}" "${destination}" >&2
    return 1
  }
  if [[ -n "${expected_sha256}" ]]; then
    actual_sha256="$(sha256sum "${destination}" | cut -d ' ' -f 1)"
    [[ "${actual_sha256}" == "${expected_sha256}" ]] || {
      printf 'Downloaded file has the wrong SHA256 (%s != %s): %s\n' \
        "${actual_sha256}" "${expected_sha256}" "${destination}" >&2
      return 1
    }
  fi
}

hf_python() {
  local candidate
  for candidate in "$(method_python refton)" "$(method_python mcld)" /home/muxiangyu/miniconda3/bin/python; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import huggingface_hub' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  printf 'No Python with huggingface_hub is available. Run setup_envs.sh first.\n' >&2
  return 1
}

hf_snapshot() {
  local repo_id="$1"
  local destination="$2"
  local python
  python="$(hf_python)"
  mkdir -p "${destination}"
  "${python}" - "${repo_id}" "${destination}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

provision_hf_dir() {
  local relative="$1"
  local repo_id="$2"
  local marker="$3"
  shift 3
  local destination="${MODEL_ROOT}/${relative}"
  local candidate
  if [[ -f "${destination}/${marker}" ]]; then
    printf 'Model ready: %s\n' "${destination}"
    return
  fi
  for candidate in "$@"; do
    if [[ -f "${candidate}/${marker}" ]]; then
      link_target "${candidate}" "${destination}"
      printf 'Linked model: %s -> %s\n' "${destination}" "${candidate}"
      return
    fi
  done
  candidate="${MODEL_STORE}/hf/${repo_id}"
  if [[ ! -f "${candidate}/${marker}" ]]; then
    printf 'Downloading %s into %s\n' "${repo_id}" "${candidate}"
    hf_snapshot "${repo_id}" "${candidate}"
  fi
  [[ -f "${candidate}/${marker}" ]] || {
    printf 'Downloaded model lacks marker %s: %s\n' "${marker}" "${candidate}" >&2
    return 1
  }
  link_target "${candidate}" "${destination}"
}

provision_ip_adapter() {
  local destination="${MODEL_ROOT}/idm_vton/ip-adapter-plus_sdxl_vit-h.bin"
  local expected_bytes=1013454427
  local expected_sha256=ec70edb7cc8e769c9388d94eeaea3e4526352c9fae793a608782d1d8951fde90
  local candidate

  ip_adapter_valid() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    [[ "$(stat -Lc '%s' "${path}")" -eq "${expected_bytes}" ]] || return 1
    local actual_sha256
    actual_sha256="$(sha256sum "${path}" | cut -d ' ' -f 1)"
    [[ "${actual_sha256}" == "${expected_sha256}" ]]
  }

  if ip_adapter_valid "${destination}"; then
    return
  fi
  if [[ -L "${destination}" ]]; then
    unlink "${destination}"
  elif [[ -e "${destination}" ]]; then
    printf 'Refusing to replace invalid non-symlink IP-Adapter: %s\n' "${destination}" >&2
    return 1
  fi
  for candidate in \
    /data/muxiangyu/programs/VTON_baselines/baselines/IDM-VTON/ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/idmvton/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin; do
    if ip_adapter_valid "${candidate}"; then
      link_target "${candidate}" "${destination}"
      return
    fi
  done
  local python downloaded
  python="$(hf_python)"
  downloaded="$("${python}" - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("h94/IP-Adapter", "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"))
PY
)"
  if ! ip_adapter_valid "${downloaded}"; then
    printf 'Downloaded IP-Adapter failed size/SHA256 validation: %s\n' "${downloaded}" >&2
    return 1
  fi
  link_target "${downloaded}" "${destination}"
}

provision_dinov2() {
  local repo_store="${MODEL_STORE}/git/dinov2"
  local destination="${MODEL_ROOT}/ita_mdt/dinov2"
  local checkpoint="${TORCH_HOME}/hub/checkpoints/dinov2_vitg14_pretrain.pth"
  # Content-Length reported by the official DINOv2 release object. Checking
  # the exact size keeps interrupted multi-GB downloads resumable.
  local expected_bytes=4546108579
  local actual_bytes=0

  if [[ ! -f "${repo_store}/hubconf.py" ]]; then
    mkdir -p "$(dirname "${repo_store}")"
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "${repo_store}"
  fi
  link_target "${repo_store}" "${destination}"
  mkdir -p "$(dirname "${checkpoint}")"
  if [[ -f "${checkpoint}" ]]; then
    actual_bytes="$(stat -c '%s' "${checkpoint}")"
  fi
  if (( actual_bytes > expected_bytes )); then
    printf 'DINOv2-G checkpoint is larger than the official object (%s > %s): %s\n' \
      "${actual_bytes}" "${expected_bytes}" "${checkpoint}" >&2
    return 1
  fi
  if (( actual_bytes < expected_bytes )); then
    printf 'Downloading/resuming DINOv2-G at byte %s of %s.\n' \
      "${actual_bytes}" "${expected_bytes}"
    curl -fL --retry 5 --continue-at - \
      -o "${checkpoint}" \
      https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
  fi
  actual_bytes="$(stat -c '%s' "${checkpoint}")"
  [[ "${actual_bytes}" -eq "${expected_bytes}" ]] || {
    printf 'DINOv2-G checkpoint has the wrong size (%s != %s): %s\n' \
      "${actual_bytes}" "${expected_bytes}" "${checkpoint}" >&2
    return 1
  }
}

provision_ita_checkpoint() {
  local cached="${MODEL_STORE}/hf/jiwoohong93/ita-mdt_weights/ema_0.9999_2000000.pt"
  local destination="${MODEL_ROOT}/ita_mdt/ema_0.9999_2000000.pt"
  local expected_bytes=2930557662
  local expected_sha256=fa0ea9b2884a6c268e5a68192a0a485af909261dd358bcfe1849770a2ee5dd6f

  download_exact_file \
    "https://huggingface.co/jiwoohong93/ita-mdt_weights/resolve/main/ema_0.9999_2000000.pt?download=true" \
    "${cached}" \
    "${expected_bytes}" \
    "${expected_sha256}"
  if [[ ! -e "${destination}" ]]; then
    link_target "${cached}" "${destination}"
  fi
  [[ "$(sha256sum "${destination}" | cut -d ' ' -f 1)" == "${expected_sha256}" ]] || {
    printf 'ITA-MDT initialization checkpoint failed SHA256 validation: %s\n' "${destination}" >&2
    return 1
  }
}

provision_antelopev2() {
  local destination="${MODEL_ROOT}/mcld/insightface/models/antelopev2"
  local candidate
  if [[ -f "${destination}/glintr100.onnx" ]]; then
    return
  fi
  for candidate in \
    /data/muxiangyu/modelLibrary/PuLID/models/antelopev2 \
    /data/muxiangyu/modelLibrary/insightface/models/antelopev2; do
    if [[ -f "${candidate}/glintr100.onnx" ]]; then
      link_target "${candidate}" "${destination}"
      return
    fi
  done
  local model_parent="${MODEL_STORE}/insightface/models"
  local archive="${CACHE_ROOT}/downloads/antelopev2.zip"
  local expected_bytes=360662982
  mkdir -p "${model_parent}"
  download_exact_file \
    https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip \
    "${archive}" \
    "${expected_bytes}"
  if [[ ! -f "${model_parent}/antelopev2/glintr100.onnx" ]]; then
    unzip -n "${archive}" -d "${model_parent}"
  fi
  link_target "${model_parent}/antelopev2" "${destination}"
}

provision_mcld_controlnet() {
  local relative=mcld/control_v11p_sd15_seg
  local destination="${MODEL_ROOT}/${relative}"
  local cached="${MODEL_STORE}/hf/lllyasviel/control_v11p_sd15_seg"
  local filename=diffusion_pytorch_model.bin
  local expected_bytes=1445254969
  local expected_sha256=4397ad13ff43760023ad5f7b13969731e52c0d23cc1c609fcb4b25da2c00be0b
  local candidate

  controlnet_valid() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    [[ "$(stat -Lc '%s' "${path}")" -eq "${expected_bytes}" ]] || return 1
    [[ "$(sha256sum "${path}" | cut -d ' ' -f 1)" == "${expected_sha256}" ]]
  }

  if controlnet_valid "${destination}/${filename}"; then
    printf 'Model ready: %s\n' "${destination}"
    return
  fi
  for candidate in \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/mcld/control_v11p_sd15_seg; do
    if controlnet_valid "${candidate}/${filename}"; then
      link_target "${candidate}" "${destination}"
      printf 'Linked model: %s -> %s\n' "${destination}" "${candidate}"
      return
    fi
  done

  download_exact_file \
    "${M2H_HF_MIRROR:-https://hf-mirror.com}/lllyasviel/control_v11p_sd15_seg/resolve/main/${filename}" \
    "${cached}/${filename}" \
    "${expected_bytes}" \
    "${expected_sha256}"
  link_target "${cached}" "${destination}"
  printf 'Model ready: %s\n' "${destination}"
}

if wants refton; then
  provision_hf_dir refton/FLUX.1-Kontext-dev black-forest-labs/FLUX.1-Kontext-dev transformer/config.json \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/refton/FLUX.1-Kontext-dev \
    /data/muxiangyu/programs/M2HImage/models/hf/black-forest-labs/FLUX.1-Kontext-dev
fi

if wants idm_vton || wants ita_mdt; then
  provision_hf_dir idm_vton/IDM-VTON yisol/IDM-VTON unet/config.json \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/idmvton
  provision_ip_adapter
fi

if wants ita_mdt; then
  link_target "${MODEL_ROOT}/idm_vton/IDM-VTON" "${MODEL_ROOT}/ita_mdt/sdxl-inpainting"
  provision_dinov2
  provision_ita_checkpoint
fi

if wants ominicontrol; then
  provision_hf_dir ominicontrol/FLUX.1-dev black-forest-labs/FLUX.1-dev transformer/config.json \
    /data/muxiangyu/programs/M2HImage/models/hf/black-forest-labs/FLUX.1-dev \
    /data/muxiangyu/pythonPrograms/M2HImage/models/hf/black-forest-labs/FLUX.1-dev
fi

if wants mcld; then
  provision_hf_dir mcld/sd-image-variations-diffusers lambdalabs/sd-image-variations-diffusers unet/config.json \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/mcld/sd-image-variations-diffusers
  provision_hf_dir mcld/sd-vae-ft-mse stabilityai/sd-vae-ft-mse config.json \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/mcld/sd-vae-ft-mse
  provision_hf_dir mcld/stable-diffusion-v1-5 stable-diffusion-v1-5/stable-diffusion-v1-5 unet/config.json \
    /data/muxiangyu/programs/VTON_baselines/checkpoints/mcld/stable-diffusion-v1-5
  provision_mcld_controlnet
  provision_antelopev2
fi

if [[ "${requested}" != "all" ]]; then
  require_method "${requested}"
fi

printf 'Model setup complete for %s. Large files live under %s or existing /data stores.\n' \
  "${requested}" "${MODEL_STORE}"
