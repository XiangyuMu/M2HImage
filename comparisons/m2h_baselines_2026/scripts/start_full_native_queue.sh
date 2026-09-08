#!/usr/bin/env bash

# Wait for the separately launched full MCLD face-staging job, require complete
# feature coverage, run both preflight contracts, then hand off to the durable
# five-method queue.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FULL_ENV="${M2H_FULL_ENV:-${PROJECT_ROOT}/configs/task_full_native_20260823.env}"
# shellcheck disable=SC1090
source "${FULL_ENV}"
source "${SCRIPT_DIR}/lib.sh"

face_session="${M2H_FACE_SESSION:-m2h_full_mcld_faces}"
while tmux has-session -t "${face_session}" 2>/dev/null; do
  printf '[%s] waiting for %s\n' "$(date +%FT%T)" "${face_session}"
  sleep 30
done

python="$(method_python mcld)"
paired="${LAYOUT_ROOT}/low/paired/mcld"
counterfactual="${LAYOUT_ROOT}/low/counterfactual/mcld"
env "PYTHONPATH=${PROJECT_ROOT}" "${python}" - "${paired}" "${counterfactual}" <<'PY'
import json
import sys
from pathlib import Path

paired, counterfactual = map(Path, sys.argv[1:])
checks = [
    (paired / "train_face", 36033),
    (paired / "test_face", 1996),
    (counterfactual / "test_face", 400),
]
for directory, expected in checks:
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    available = len(list(directory.glob("*.npy")))
    if payload.get("failures"):
        raise RuntimeError(f"{directory}: face failures={len(payload['failures'])}")
    if available != expected or int(payload.get("available", -1)) != expected:
        raise RuntimeError(
            f"{directory}: embedding coverage={available}/{expected}, "
            f"manifest={payload.get('available')}"
        )
    print(f"face coverage pass: {directory} {available}/{expected}")
PY

preflight_root="${RUN_ROOT}/orchestrator/preflight"
mkdir -p "${preflight_root}"
"${SCRIPT_DIR}/preflight.sh" all paired full 2>&1 | tee "${preflight_root}/paired.log"
"${SCRIPT_DIR}/preflight.sh" all counterfactual full 2>&1 | tee "${preflight_root}/counterfactual.log"

exec "${SCRIPT_DIR}/run_full_native.sh" --skip-prepare
