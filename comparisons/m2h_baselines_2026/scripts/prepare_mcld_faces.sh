#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

kind="${1:-all}"
case "${kind}" in
  all|paired|counterfactual) ;;
  *) printf 'Kind must be all, paired, or counterfactual: %s\n' "${kind}" >&2; exit 2 ;;
esac

python="$(method_python mcld)"
model_root="${MODEL_ROOT}/mcld/insightface"
require_file "${python}"
require_file "${model_root}/models/antelopev2/glintr100.onnx"

limit_args=()
if [[ -n "${M2H_FACE_LIMIT:-}" ]]; then
  limit_args=(--limit "${M2H_FACE_LIMIT}")
fi

face_device="${M2H_FACE_DEVICE:-cpu}"
runtime_env=("PYTHONPATH=${PROJECT_ROOT}")
if [[ "${face_device}" == "cuda" ]]; then
  site_packages="$("${python}" -c 'import site; print(site.getsitepackages()[0])')"
  torch_lib="${site_packages}/torch/lib"
  nvidia_root="${M2H_CUDA11_NVIDIA_ROOT:-/home/muxiangyu/miniconda3/envs/gsinpaint/lib/python3.10/site-packages/nvidia}"
  require_dir "${torch_lib}"
  require_dir "${nvidia_root}/cublas/lib"
  require_dir "${nvidia_root}/cufft/lib"
  runtime_env+=(
    "CUDA_VISIBLE_DEVICES=${M2H_FACE_GPU:-0}"
    "LD_LIBRARY_PATH=${torch_lib}:${nvidia_root}/cublas/lib:${nvidia_root}/cuda_runtime/lib:${nvidia_root}/cudnn/lib:${nvidia_root}/cufft/lib:${nvidia_root}/curand/lib:${nvidia_root}/cusolver/lib:${nvidia_root}/cusparse/lib:${nvidia_root}/nccl/lib:${nvidia_root}/nvtx/lib:${LD_LIBRARY_PATH:-}"
  )
fi

build_phase() {
  local layout="$1"
  local phase="$2"
  require_dir "${layout}/${phase}_face_inputs"
  run_logged mcld faces env "${runtime_env[@]}" \
    "${python}" -m m2h_baselines.mcld_face \
    --layout-root "${layout}" \
    --phase "${phase}" \
    --model-root "${model_root}" \
    --device "${face_device}" \
    "${limit_args[@]}"
}

if [[ "${kind}" == "all" || "${kind}" == "paired" ]]; then
  paired="${LAYOUT_ROOT}/low/paired/mcld"
  build_phase "${paired}" train
  build_phase "${paired}" test
fi

if [[ "${kind}" == "all" || "${kind}" == "counterfactual" ]]; then
  build_phase "${LAYOUT_ROOT}/low/counterfactual/mcld" test
fi

printf 'MCLD Antelopev2 face embeddings are ready for %s.\n' "${kind}"
