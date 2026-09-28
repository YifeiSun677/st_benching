#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # → st_benching root
export PYTHONPATH=.

echo "==== preflight ===="
python -m path2space.preflight

echo "==== build full feature cache (one-time) ===="
python -m path2space.build_features

echo "==== full LOPO ensemble training ===="
python -m path2space.run_lopo --tag path2space_lopo_833

echo "==== scoring (raw + smoothed variant) ===="
python -m path2space.score --tag path2space_lopo_833
python -m path2space.score --tag path2space_lopo_833 --smooth
