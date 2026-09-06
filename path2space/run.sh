#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # → st_benching 根
export PYTHONPATH=.

echo "==== preflight ===="
python -m path2space.preflight

echo "==== 构建全量特征缓存（一次性）===="
python -m path2space.build_features

echo "==== 完整 LOPO 集成训练 ===="
python -m path2space.run_lopo --tag path2space_lopo_833

echo "==== 打分（raw + 平滑变体）===="
python -m path2space.score --tag path2space_lopo_833
python -m path2space.score --tag path2space_lopo_833 --smooth
