#!/usr/bin/env bash
set -euo pipefail
python -m mcic.preprocess.audit --config configs/preprocess.yaml "$@"
