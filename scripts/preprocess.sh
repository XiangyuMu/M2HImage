#!/usr/bin/env bash
set -euo pipefail
python -m mcic.preprocess.run --config configs/preprocess.yaml "$@"
