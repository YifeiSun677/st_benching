#!/usr/bin/env bash
# Environment for HisToGene on a RunPod pod. Run with:  source scripts/00_env.sh
set -euo pipefail

export HTG_WORKSPACE=/workspace
export HTG_HER2ST=/workspace/her2st/data
export HTG_REPO=/workspace/HisToGene
export HTG_CACHE=/workspace/cache/htg_patch112
export HTG_OUT=/workspace/runs
export HTG_PANEL=/workspace/histogene-her2st/panels/panel_833.txt

# keep pip/HF caches on the network volume so a pod restart does not re-download
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

mkdir -p "$HTG_CACHE" "$HTG_OUT" "$PIP_CACHE_DIR"
echo "env set. workspace=$HTG_WORKSPACE  panel=$HTG_PANEL"
