#!/usr/bin/env bash
# HisToGene environment for a RunPod pod.
# Run from the st_benching repo root:   source histogene/env.sh
export HTG_WORKSPACE=/workspace
export HTG_HER2ST=/workspace/her2st/data
export HTG_REPO=/workspace/HisToGene            # clone of maxpmx/HisToGene
export HTG_CACHE=/workspace/cache/htg_patch112
export HTG_OUT=/workspace/runs                  # large: preds + ckpts, not in git

# panel / results / gene sets default to the st_benching repo, no need to set.
# export HTG_PANEL=<repo>/panels/panel_833.txt
# export HTG_RESULTS=<repo>/results
# export HTG_GENE_SETS=<repo>/results/gene_sets

export PIP_CACHE_DIR=/workspace/.cache/pip
export TORCH_HOME=/workspace/.cache/torch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

mkdir -p "$HTG_CACHE" "$HTG_OUT" "$PIP_CACHE_DIR"
echo "HisToGene env set. cache=$HTG_CACHE out=$HTG_OUT"
