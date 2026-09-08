#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="${1:-smoke}"
resolution="${2:-low}"
require_profile "${profile}"
case "${resolution}" in
  low|native) ;;
  *) printf 'Resolution must be low or native, got: %s\n' "${resolution}" >&2; exit 2 ;;
esac

prep_python="${M2H_PREP_PYTHON:-$(method_python refton)}"
if [[ ! -x "${prep_python}" ]]; then
  prep_python="/home/muxiangyu/miniconda3/bin/python"
fi
require_file "${DATA_ROOT}/splits/train.txt"
require_file "${DATA_ROOT}/splits/val.txt"
require_file "${DATA_ROOT}/eval/cf_subset.json"
require_file "${CONFIG_PATH}"

common=(
  env "PYTHONPATH=${PROJECT_ROOT}"
  "${prep_python}" -m m2h_baselines.cli
  --config "${CONFIG_PATH}"
)
train_limit=()
val_limit=()
cf_limit=()
if [[ "${profile}" == "smoke" ]]; then
  train_limit=(--limit 32)
  val_limit=(--limit 8)
  cf_limit=(--limit-pairs 2)
fi

run_logged data "${profile}" "${common[@]}" prepare \
  --split train \
  --resolution "${resolution}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --workers "${M2H_PREP_WORKERS:-12}" \
  "${train_limit[@]}"

run_logged data "${profile}" "${common[@]}" prepare \
  --split val \
  --resolution "${resolution}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --workers "${M2H_PREP_WORKERS:-12}" \
  "${val_limit[@]}"

run_logged data "${profile}" "${common[@]}" prepare-cf \
  --resolution "${resolution}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${PREPARED_ROOT}" \
  --protocol "${DATA_ROOT}/eval/cf_subset.json" \
  --workers "${M2H_PREP_WORKERS:-12}" \
  "${cf_limit[@]}"

for split in train val counterfactual; do
  manifest="${PREPARED_ROOT}/${resolution}/${split}/samples.jsonl"
  run_logged data "${profile}" "${common[@]}" validate \
    --manifest "${manifest}" \
    --prepared-root "${PREPARED_ROOT}"
done

paired_layout="${LAYOUT_ROOT}/${resolution}/paired"
counterfactual_layout="${LAYOUT_ROOT}/${resolution}/counterfactual"
for method in refton ita_mdt idm_vton mcld; do
  for split in train val; do
    run_logged data "${profile}" "${common[@]}" materialize \
      --method "${method}" \
      --manifest "${PREPARED_ROOT}/${resolution}/${split}/samples.jsonl" \
      --prepared-root "${PREPARED_ROOT}" \
      --output-root "${paired_layout}"
  done
  run_logged data "${profile}" "${common[@]}" materialize \
    --method "${method}" \
    --manifest "${PREPARED_ROOT}/${resolution}/counterfactual/samples.jsonl" \
    --prepared-root "${PREPARED_ROOT}" \
    --output-root "${counterfactual_layout}"
done

broken_link="$(find -L "${LAYOUT_ROOT}/${resolution}" -type l -print -quit)"
if [[ -n "${broken_link}" ]]; then
  printf 'Broken layout symlink: %s\n' "${broken_link}" >&2
  exit 1
fi

printf 'Prepared profile=%s resolution=%s under %s\n' \
  "${profile}" "${resolution}" "${PREPARED_ROOT}"
