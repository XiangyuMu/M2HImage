#!/usr/bin/env bash
set -euo pipefail
accelerate launch --num_processes 2 -m mcic.training.train --config configs/exp02_cf_cloth.yaml "$@"
