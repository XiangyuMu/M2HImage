#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

profile="${1:-full}"
[[ "${profile}" == "full" ]] || {
  printf 'This helper only prepares the full profile, got %s\n' "${profile}" >&2
  exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build both views from the same source dataset.  Native is the frozen protocol
# view; low is the deterministic 512x512 model staging view.
"${script_dir}/prepare_data.sh" full native
"${script_dir}/prepare_data.sh" full low

native_train="${PREPARED_ROOT}/native/train/samples.jsonl"
low_train="${PREPARED_ROOT}/low/train/samples.jsonl"
native_val="${PREPARED_ROOT}/native/val/samples.jsonl"
low_val="${PREPARED_ROOT}/low/val/samples.jsonl"
native_cf="${PREPARED_ROOT}/native/counterfactual/samples.jsonl"
low_cf="${PREPARED_ROOT}/low/counterfactual/samples.jsonl"
for manifest in "${native_train}" "${low_train}" "${native_val}" "${low_val}" "${native_cf}" "${low_cf}"; do
  require_file "${manifest}"
done

native_train_count="$(wc -l < "${native_train}")"
low_train_count="$(wc -l < "${low_train}")"
native_val_count="$(wc -l < "${native_val}")"
low_val_count="$(wc -l < "${low_val}")"
native_cf_count="$(wc -l < "${native_cf}")"
low_cf_count="$(wc -l < "${low_cf}")"
if [[ "${native_train_count}" != "${low_train_count}" || \
      "${native_val_count}" != "${low_val_count}" || \
      "${native_cf_count}" != "${low_cf_count}" ]]; then
  printf 'Native/low manifest count mismatch: train=%s/%s val=%s/%s cf=%s/%s\n' \
    "${native_train_count}" "${low_train_count}" \
    "${native_val_count}" "${low_val_count}" \
    "${native_cf_count}" "${low_cf_count}" >&2
  exit 1
fi

printf 'Full native preparation passed: train=%s val=%s counterfactual=%s\n' \
  "${native_train_count}" "${native_val_count}" "${native_cf_count}"
