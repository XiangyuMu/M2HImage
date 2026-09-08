#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

requested="${1:-all}"
base_python="${M2H_BASE_PYTHON:-/home/muxiangyu/miniconda3/bin/python}"
pack_tool_root="${M2H_CONDA_PACK_TOOL_ROOT:-${CACHE_ROOT}/tools/conda-pack}"
pack_root="${M2H_CONDA_PACK_ROOT:-${CACHE_ROOT}/conda-packs}"
pack_version="${M2H_CONDA_PACK_VERSION:-0.8.1}"
rebuild_packs="${M2H_REBUILD_PACKS:-0}"

require_file "${base_python}"
require_command tar
mkdir -p "${ENV_ROOT}" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/tmp" \
  "${pack_tool_root}" "${pack_root}"

wants() {
  [[ "${requested}" == "all" || "${requested}" == "$1" ]]
}

ensure_conda_pack() {
  if PYTHONPATH="${pack_tool_root}" "${base_python}" -c \
      'import conda_pack' >/dev/null 2>&1; then
    return
  fi

  printf 'Installing conda-pack %s under %s\n' "${pack_version}" "${pack_tool_root}" >&2
  "${base_python}" -m pip install \
    --target "${pack_tool_root}" \
    --upgrade \
    'setuptools<81' \
    "conda-pack==${pack_version}" >&2
  PYTHONPATH="${pack_tool_root}" "${base_python}" -c \
    'import conda_pack; print("conda-pack", conda_pack.__version__)' >&2
}

build_pack() {
  local pack_name="$1"
  local source_prefix="$2"
  local archive="${pack_root}/${pack_name}.tar.gz"
  local partial="${pack_root}/.${pack_name}.partial.tar.gz"

  require_file "${source_prefix}/bin/python"
  require_dir "${source_prefix}/conda-meta"

  if [[ "${rebuild_packs}" != "1" && -s "${archive}" ]]; then
    printf 'Reusing environment archive: %s\n' "${archive}" >&2
    printf '%s\n' "${archive}"
    return
  fi

  ensure_conda_pack
  printf 'Packing %s into %s\n' "${source_prefix}" "${archive}" >&2
  rm -f "${partial}"
  PYTHONPATH="${pack_tool_root}" "${base_python}" - "${source_prefix}" "${partial}" <<'PY'
import sys
from conda_pack import pack

pack(
    prefix=sys.argv[1],
    output=sys.argv[2],
    format="tar.gz",
    force=True,
    ignore_editable_packages=True,
    ignore_missing_files=True,
    n_threads=4,
)
PY
  mv "${partial}" "${archive}"
  printf '%s\n' "${archive}"
}

verify_env() {
  local target_prefix="$1"
  require_file "${target_prefix}/bin/python"
  require_dir "${target_prefix}/conda-meta"
  "${target_prefix}/bin/python" - "${target_prefix}" <<'PY'
import os
import sys

expected = os.path.realpath(sys.argv[1])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise SystemExit(f"unexpected sys.prefix: {actual} != {expected}")
import torch
import diffusers
print(f"verified {expected}: python={sys.version.split()[0]} torch={torch.__version__} diffusers={diffusers.__version__}")
PY
}

install_packed_env() {
  local method="$1"
  local pack_name="$2"
  local source_prefix="$3"
  local target_prefix="${ENV_ROOT}/${method}"
  local ready_marker="${target_prefix}/.m2h_conda_pack_ready"
  local archive

  if [[ -f "${ready_marker}" ]]; then
    verify_env "${target_prefix}"
    printf 'Environment exists: %s\n' "${target_prefix}"
    return
  fi

  if [[ -e "${target_prefix}" ]]; then
    if [[ -x "${target_prefix}/bin/conda-unpack" && \
          -x "${target_prefix}/bin/python" && \
          -d "${target_prefix}/conda-meta" ]]; then
      printf 'Finishing previously extracted environment: %s\n' "${target_prefix}"
      "${target_prefix}/bin/python" "${target_prefix}/bin/conda-unpack"
      verify_env "${target_prefix}"
      touch "${ready_marker}"
      return
    fi
    printf 'Incomplete environment exists: %s\n' "${target_prefix}" >&2
    printf 'Refusing to overwrite it. Verify this exact path and remove it before retrying.\n' >&2
    return 1
  fi

  archive="$(build_pack "${pack_name}" "${source_prefix}")"
  printf 'Extracting %s into %s\n' "${archive}" "${target_prefix}"
  mkdir -p "${target_prefix}"
  if ! tar -xzf "${archive}" -C "${target_prefix}"; then
    printf 'Extraction failed; incomplete target left at %s\n' "${target_prefix}" >&2
    return 1
  fi
  require_file "${target_prefix}/bin/conda-unpack"
  "${target_prefix}/bin/python" "${target_prefix}/bin/conda-unpack"
  verify_env "${target_prefix}"
  touch "${ready_marker}"
}

pip_install() {
  local method="$1"
  shift
  "${ENV_ROOT}/${method}/bin/python" -m pip install "$@"
}

install_xformers() {
  local method="$1"
  local wheel="${CACHE_ROOT}/wheels/xformers-0.0.22.post7-cp310-cp310-manylinux2014_x86_64.whl"
  if [[ -s "${wheel}" ]]; then
    printf 'Installing cached xformers wheel: %s\n' "${wheel}"
    pip_install "${method}" --no-deps "${wheel}"
  else
    pip_install "${method}" --no-deps xformers==0.0.22.post7
  fi
}

install_project() {
  local method="$1"
  pip_install "${method}" --no-deps -e "${PROJECT_ROOT}"
}

case "${requested}" in
  all|refton|ita_mdt|idm_vton|mcld|ominicontrol) ;;
  *) printf 'Unknown method: %s\n' "${requested}" >&2; exit 2 ;;
esac

if wants refton; then
  install_packed_env refton refton /home/muxiangyu/miniconda3/envs/refton
  install_project refton
fi

if wants idm_vton; then
  install_packed_env idm_vton idmvton /home/muxiangyu/miniconda3/envs/idmvton
  pip_install idm_vton 'bitsandbytes>=0.43,<0.46'
  pip_install idm_vton --no-deps nvidia-cusparse-cu11==11.7.5.86
  install_xformers idm_vton
  install_project idm_vton
fi

if wants ita_mdt; then
  install_packed_env ita_mdt idmvton /home/muxiangyu/miniconda3/envs/idmvton
  pip_install ita_mdt \
    numpy==1.25.2 \
    huggingface-hub==0.34.4 \
    'setuptools<81' \
    'blobfile>=1.0.5' \
    diffusers==0.34.0 \
    accelerate==1.0.1 \
    timm==0.9.16 \
    einops==0.8.0 \
    albumentations==1.4.24 \
    opencv-python-headless==4.10.0.84 \
    'scikit-image>=0.21' \
    ninja
  "${ENV_ROOT}/ita_mdt/bin/python" -m pip uninstall -y opencv-python
  pip_install ita_mdt --force-reinstall --no-deps opencv-python-headless==4.10.0.84
  env CUDA_VISIBLE_DEVICES="" \
    "${ENV_ROOT}/ita_mdt/bin/python" -m pip install --no-build-isolation \
    'git+https://github.com/sail-sg/Adan.git'
  pip_install ita_mdt --no-deps -e "${PROJECT_ROOT}/repos/ITA-MDT"
  install_project ita_mdt
  "${ENV_ROOT}/ita_mdt/bin/python" - <<'PY'
import albumentations
import cv2
import diffusers
import huggingface_hub
import numpy
import torch
import transformers
from adan import Adan

assert numpy.__version__ == "1.25.2"
assert huggingface_hub.__version__ == "0.34.4"
print("ITA-MDT imports verified")
PY
fi

if wants mcld; then
  install_packed_env mcld idmvton /home/muxiangyu/miniconda3/envs/idmvton
  pip_install mcld \
    numpy==1.25.2 \
    'setuptools<81' \
    wheel==0.42.0 \
    accelerate==0.23.0 \
    diffusers==0.24.0 \
    einops==0.4.1 \
    transformers==4.46.3 \
    huggingface_hub==0.25.0 \
    mlflow==2.9.2 \
    lpips \
    torchdiffeq==0.2.3 \
    torchmetrics==1.2.1 \
    torchsde \
    omegaconf==2.2.3 \
    albumentations==1.3.1 \
    opencv-python-headless==4.8.1.78 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.2 \
    pandas \
    uvtextureconverter==1.2.0 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.16.3 \
    'bitsandbytes>=0.43,<0.46'
  "${ENV_ROOT}/mcld/bin/python" -m pip uninstall -y opencv-python
  pip_install mcld --force-reinstall --no-deps opencv-python-headless==4.8.1.78
  install_xformers mcld
  install_project mcld
  pushd "${PROJECT_ROOT}/repos/MCLD" >/dev/null
  "${ENV_ROOT}/mcld/bin/python" - <<'PY'
import accelerate, cv2, diffusers, insightface, mlflow, numpy, onnxruntime, torch, transformers, xformers
from src.dataset.deepfashion_dataset import get_deepfashion_dataset
assert numpy.__version__ == "1.25.2"
print("MCLD imports verified", cv2.__version__, diffusers.__version__, transformers.__version__)
PY
  popd >/dev/null
fi

if wants ominicontrol; then
  install_packed_env ominicontrol refton /home/muxiangyu/miniconda3/envs/refton
  pip_install ominicontrol \
    transformers==4.57.6 \
    huggingface_hub==0.36.2 \
    lightning==2.5.2 \
    prodigyopt
  install_project ominicontrol
  pushd "${PROJECT_ROOT}/repos/OminiControl" >/dev/null
  "${ENV_ROOT}/ominicontrol/bin/python" - <<'PY'
import cv2, diffusers, huggingface_hub, lightning, peft, torch, transformers
from omini.train_flux.train_m2h import M2HManifestDataset
assert transformers.__version__ == "4.57.6"
assert huggingface_hub.__version__ == "0.36.2"
print("OminiControl imports verified")
PY
  popd >/dev/null
fi

printf 'Environment setup complete for: %s\n' "${requested}"
