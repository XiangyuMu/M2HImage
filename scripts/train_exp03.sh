#!/usr/bin/env bash
set -euo pipefail
accelerate launch --num_processes 2 -m mcic.training.train --config configs/exp03_cf_identity.yaml "$@"
