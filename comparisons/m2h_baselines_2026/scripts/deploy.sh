#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

remote="${M2H_REMOTE:-muxiangyu@10.249.190.76}"
remote_root="${M2H_REMOTE_ROOT:-/home/muxiangyu/programs/M2H_Baselines_2026}"
dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=(--dry-run)
elif [[ -n "${1:-}" ]]; then
  remote="$1"
fi

require_command ssh
require_command rsync
ssh "${remote}" "mkdir -p '${remote_root}' '/data/muxiangyu/envs/m2h_baselines_2026' '/data/muxiangyu/cache/m2h_baselines_2026'"
rsync -az --links \
  "${dry_run[@]}" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'prepared*/' \
  --exclude 'layouts*/' \
  --exclude 'runs/' \
  --exclude 'outputs/' \
  --exclude 'reports/generated/' \
  "${PROJECT_ROOT}/" "${remote}:${remote_root}/"
ssh "${remote}" "find '${remote_root}/scripts' -type f -name '*.sh' -exec chmod +x {} +"
printf 'Deployed code to %s:%s (no remote files were deleted).\n' "${remote}" "${remote_root}"
