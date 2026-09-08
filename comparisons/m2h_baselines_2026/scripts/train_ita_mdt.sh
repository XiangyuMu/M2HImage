#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="$(profile_from_invocation "${1:-}")"
require_profile "${profile}"
repo="${PROJECT_ROOT}/repos/ITA-MDT"
python="$(method_python ita_mdt)"
data_dir="${LAYOUT_ROOT}/low/paired/ita_mdt"
vae_model="${MODEL_ROOT}/ita_mdt/sdxl-inpainting"
dino_repo="${M2H_DINO_REPO:-${MODEL_ROOT}/ita_mdt/dinov2}"
init_checkpoint="${M2H_ITA_INIT_CHECKPOINT:-${MODEL_ROOT}/ita_mdt/ema_0.9999_2000000.pt}"
resume_checkpoint="${M2H_ITA_RESUME_CHECKPOINT:-}"
output_dir="${RUN_ROOT}/ita_mdt/${profile}"
steps="${M2H_STEPS:-200000}"
save_interval="${M2H_CHECKPOINT_STEPS:-10000}"
if [[ "${profile}" == "smoke" ]]; then
  steps="${M2H_STEPS:-100}"
  save_interval="${M2H_CHECKPOINT_STEPS:-100}"
fi
require_file "${python}"
require_dir "${data_dir}/zalando-hd-resized/train"
require_dir "${vae_model}/vae"
require_dir "${dino_repo}"
checkpoint_args=(--init_checkpoint "${init_checkpoint}")
if [[ -n "${resume_checkpoint}" ]]; then
  require_file "${resume_checkpoint}"
  checkpoint_args=(--resume_checkpoint "${resume_checkpoint}")
else
  require_file "${init_checkpoint}"
fi
require_file "${TORCH_HOME}/hub/checkpoints/dinov2_vitg14_pretrain.pth"
mkdir -p "${output_dir}"

pushd "${repo}" >/dev/null
run_logged ita_mdt "${profile}" env \
  "OPENAI_LOGDIR=${output_dir}" \
  "${python}" -m torch.distributed.run \
  --nproc_per_node "${M2H_NUM_PROCESSES}" \
  --master_port "${M2H_PORT:-29532}" \
  image_train.py \
  --data_dir "${data_dir}" \
  --dataset_mode m2h \
  --work_dir "${output_dir}" \
  --image_size 512 \
  --vit_image_size 224 \
  --mask_ratio 0.30 \
  --decode_layer 4 \
  --model MDT_IVTON_XL \
  --diffusion_steps 1000 \
  --batch_size "${M2H_GLOBAL_BATCH:-2}" \
  --microbatch 1 \
  --n_gpus "${M2H_NUM_PROCESSES}" \
  --lr "${M2H_LR:-1e-4}" \
  --lr_anneal_steps "${steps}" \
  --save_interval "${save_interval}" \
  --log_interval 1 \
  --use_fp16 "${M2H_ITA_USE_FP16:-False}" \
  --gradient_checkpointing "${M2H_GRADIENT_CHECKPOINTING:-True}" \
  --num_workers "${M2H_DATALOADER_WORKERS:-4}" \
  --seed 42 \
  --transform_size "shiftscale hflip" \
  --transform_color "hsv bright_contrast" \
  --vae_model_path "${vae_model}" \
  --vae_subfolder vae \
  --dino_repo_or_dir "${dino_repo}" \
  --dino_model dinov2_vitg14 \
  "${checkpoint_args[@]}"
popd >/dev/null

env "PYTHONPATH=${PROJECT_ROOT}" "${python}" -m m2h_baselines.record_checkpoint \
  --method ita_mdt \
  --run-dir "${output_dir}" \
  --protocol-sha256 "${PROTOCOL_SHA256}" \
  --base-model "${init_checkpoint}" \
  --data-dir "${data_dir}"
