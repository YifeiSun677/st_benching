#!/usr/bin/env bash
set -euo pipefail
export HER2ST_DIR=${HER2ST_DIR:-/workspace/her2st/data}
export HIST2ST_CACHE=${HIST2ST_CACHE:-/workspace/her2st_cache_112}
export HIST2ST_REPO=${HIST2ST_REPO:-/workspace/repos/Hist2ST}
export HIST2ST_OUT=${HIST2ST_OUT:-/workspace/runs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/st_benching
python -m hist2st.train --tag hist2st_lopo_833 --cv patient --folds all \
       --epochs 350 --lr 1e-5 --save_ckpt --log_every 25 2>&1 \
  | tee -a "${HIST2ST_OUT}/hist2st_lopo_833.log"
python -m hist2st.score --tag hist2st_lopo_833
