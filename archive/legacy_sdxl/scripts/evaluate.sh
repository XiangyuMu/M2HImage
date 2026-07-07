#!/usr/bin/env bash
set -euo pipefail
python -m mcic.evaluation.evaluate --config configs/eval.yaml "$@"
