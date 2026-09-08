#!/usr/bin/env bash

# Independent finalizer for the months-long full queue.  It never signals or
# restarts the training session; it only waits for that session to disappear,
# then snapshots all terminal phase/artifact states into the final comparison.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FULL_ENV="${M2H_FULL_ENV:-${PROJECT_ROOT}/configs/task_full_native_20260823.env}"
# shellcheck disable=SC1090
source "${FULL_ENV}"

queue_session="${M2H_FULL_QUEUE_SESSION:-m2h_full_native_queue}"
poll_seconds="${M2H_FINALIZER_POLL_SECONDS:-300}"
case "${poll_seconds}" in
  ''|*[!0-9]*)
    printf 'M2H_FINALIZER_POLL_SECONDS must be a positive integer, got %q\n' "${poll_seconds}" >&2
    exit 2
    ;;
esac
if [[ "${poll_seconds}" -lt 1 ]]; then
  printf 'M2H_FINALIZER_POLL_SECONDS must be positive\n' >&2
  exit 2
fi

printf '[%s] finalizer waiting for tmux session %s\n' "$(date +%FT%T%z)" "${queue_session}"
while tmux has-session -t "${queue_session}" 2>/dev/null; do
  sleep "${poll_seconds}"
done

printf '[%s] queue session ended; writing strict final summary\n' "$(date +%FT%T%z)"
"${SCRIPT_DIR}/summarize_full_native.sh" --require-terminal
