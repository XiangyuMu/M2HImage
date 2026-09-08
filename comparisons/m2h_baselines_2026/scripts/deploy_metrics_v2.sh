#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

remote="${M2H_REMOTE:-muxiangyu@10.249.190.76}"
remote_main_root="${M2H_MAIN_REPO:-/data/muxiangyu/programs/M2HImage}"
local_main_root="${M2H_LOCAL_MAIN_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
remote_model_root="${M2H_METRICS_MODEL_ROOT:-/data/muxiangyu/cache/m2h_baselines_2026/models}"
metrics_python="${M2H_METRICS_PYTHON:-/data/muxiangyu/envs/m2h_baselines_2026/mcld/bin/python}"
fashn_source="${M2H_FASHN_SOURCE:-${local_main_root}/models/hf/fashn-ai/fashn-human-parser}"
adaface_source="${M2H_ADAFACE_SOURCE:-/data/muxiangyu/modelLibrary/uniface/adaface_ir_101.onnx}"
expected_adaface_sha256="f2eb07d03de0af560a82e1214df799fec5e09375d43521e2868f9dc387e5a43e"

require_command ssh
require_command rsync
require_file "${local_main_root}/conditions.py"
require_file "${local_main_root}/eval_metrics_v2.py"
require_file "${local_main_root}/configs/warmup.yaml"
require_file "${local_main_root}/configs/metrics_v2.yaml"
require_dir "${local_main_root}/metrics"
require_dir "${local_main_root}/metrics_v2"
require_dir "${fashn_source}"
require_file "${fashn_source}/model.safetensors"
require_file "${adaface_source}"

actual_adaface_sha256="$(sha256sum "${adaface_source}" | awk '{print $1}')"
if [[ "${actual_adaface_sha256}" != "${expected_adaface_sha256}" ]]; then
  printf 'AdaFace hash mismatch: expected %s, got %s\n' \
    "${expected_adaface_sha256}" "${actual_adaface_sha256}" >&2
  exit 1
fi

ssh "${remote}" mkdir -p \
  "${remote_main_root}/configs" \
  "${remote_main_root}/metrics" \
  "${remote_main_root}/metrics_v2" \
  "${remote_main_root}/models/hf/fashn-ai/fashn-human-parser" \
  "${remote_model_root}/uniface"

rsync -az \
  "${local_main_root}/conditions.py" \
  "${local_main_root}/eval_metrics_v2.py" \
  "${remote}:${remote_main_root}/"
rsync -az "${local_main_root}/metrics/" "${remote}:${remote_main_root}/metrics/"
rsync -az "${local_main_root}/metrics_v2/" "${remote}:${remote_main_root}/metrics_v2/"
rsync -az \
  "${local_main_root}/configs/warmup.yaml" \
  "${local_main_root}/configs/metrics_v2.yaml" \
  "${remote}:${remote_main_root}/configs/"
rsync -az "${fashn_source}/" \
  "${remote}:${remote_main_root}/models/hf/fashn-ai/fashn-human-parser/"
rsync -az "${adaface_source}" "${remote}:${remote_model_root}/uniface/"

ssh "${remote}" env \
  PIP_CACHE_DIR=/data/muxiangyu/cache/m2h_baselines_2026/pip \
  "${metrics_python}" -m pip install --no-deps uniface==3.7.1

remote_hash="$(ssh "${remote}" sha256sum "${remote_model_root}/uniface/adaface_ir_101.onnx" | awk '{print $1}')"
if [[ "${remote_hash}" != "${expected_adaface_sha256}" ]]; then
  printf 'Remote AdaFace hash mismatch: expected %s, got %s\n' \
    "${expected_adaface_sha256}" "${remote_hash}" >&2
  exit 1
fi

ssh "${remote}" \
  "env PYTHONPATH='${remote_main_root}' '${metrics_python}' -c 'import conditions, metrics_v2, uniface; print(\"Metrics-v2 imports: ok; UniFace\", uniface.__version__)'"

printf 'Deployed Metrics-v2 code and evaluation weights to %s (no remote files were deleted).\n' "${remote}"
