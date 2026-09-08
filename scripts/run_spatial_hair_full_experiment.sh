#!/usr/bin/env bash
set -euo pipefail

cd /data/muxiangyu/pythonPrograms/M2HImage

scripts/run_spatial_hair_ab.sh all
scripts/run_spatial_directed_ab.sh all
