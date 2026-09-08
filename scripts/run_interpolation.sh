#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python}
CONFIG=${CONFIG:-configs/interpolation.yaml}

# Smoke protocols: one alpha over five mids, then both checkpoints over two
# identity paths. Existing full-evaluation outputs are untouched.
"$PYTHON" run_interpolation.py --config "$CONFIG" --part weights --stage generate --smoke --overwrite
"$PYTHON" run_interpolation.py --config "$CONFIG" --part identity --stage generate --smoke --overwrite

# Full 4-GPU generation, official metrics, and both preregistered reports.
"$PYTHON" run_interpolation.py --config "$CONFIG" --part all --stage all --nproc 4

