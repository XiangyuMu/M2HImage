#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/ITA-MDT"
python="$(method_python ita_mdt)"
data_dir="${LAYOUT_ROOT}/low/counterfactual/ita_mdt"
vae_model="${MODEL_ROOT}/ita_mdt/sdxl-inpainting"
dino_repo="${M2H_DINO_REPO:-${MODEL_ROOT}/ita_mdt/dinov2}"
raw_dir="${OUTPUT_ROOT}/ita_mdt/${profile}/raw"
if [[ -n "${M2H_CHECKPOINT:-}" ]]; then
  checkpoint="${M2H_CHECKPOINT}"
else
  checkpoint="$(latest_file "${RUN_ROOT}/ita_mdt/${profile}" 'ema_0.9999_*.pt')"
fi
require_file "${python}"
require_file "${checkpoint}"
if [[ "${profile}" == "full" || -f "$(dirname "${checkpoint}")/m2h_checkpoint.json" ]]; then
  env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.verify_checkpoint \
    --method ita_mdt \
    --checkpoint "${checkpoint}" \
    --protocol-sha256 "${PROTOCOL_SHA256}"
fi
require_dir "${data_dir}/zalando-hd-resized/test"
require_dir "${vae_model}/vae"
require_dir "${dino_repo}"
require_file "${TORCH_HOME}/hub/checkpoints/dinov2_vitg14_pretrain.pth"
mkdir -p "${raw_dir}"

pushd "${repo}" >/dev/null
run_logged ita_mdt "infer_${profile}" env \
  "CUDA_VISIBLE_DEVICES=${M2H_INFER_GPU:-0}" \
  "${python}" generate_vitonhd.py \
  --data_dir "${data_dir}" \
  --output_dir "${raw_dir}" \
  --save_dir "${raw_dir}" \
  --model_path "${checkpoint}" \
  --image_size 512 \
  --vit_img_size 224 \
  --num_sampling_steps "${M2H_INFER_STEPS:-30}" \
  --cfg_scale 2.0 \
  --pow_scale 1.0 \
  --batch_size "${M2H_INFER_BATCH:-1}" \
  --num_workers "${M2H_DATALOADER_WORKERS:-2}" \
  --seed 42 \
  --unpair false \
  --vae_model_path "${vae_model}" \
  --vae_subfolder vae \
  --dino_repo_or_dir "${dino_repo}" \
  --dino_model dinov2_vitg14 \
  --repaint_erosion_width "${M2H_ITA_REPAINT_EROSION_WIDTH:-4}"
popd >/dev/null

collect_method_outputs ita_mdt "${profile}" "${checkpoint}" "${raw_dir}"
printf 'ITA-MDT canonical outputs: %s\n' "${OUTPUT_ROOT}/ita_mdt/${profile}/canonical"
