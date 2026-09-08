#!/usr/bin/env bash
# TCGN port driver. Run from st_benching/tcgn on the pod.
# Assumes torch+torchvision already present; installs the small extras.
set -euo pipefail

export TCGN_REPO_DIR="${TCGN_REPO_DIR:-/workspace/TCGN}"
export HER2ST_DIR="${HER2ST_DIR:-/workspace/her2st/data}"
export TCGN_CACHE="${TCGN_CACHE:-/workspace/tcgn_cache}"
export TCGN_OUT="${TCGN_OUT:-/workspace/tcgn_out}"
EPOCHS="${EPOCHS:-50}"

echo "== 0. extras =="
pip install -q timm==0.9.16 einops scipy memory_profiler Pillow

echo "== 1. preflight =="
python preflight.py

echo "== 2. build 112px patch cache (once) =="
python patch_cache.py

echo "== 3. timing probe: 2 epochs on one LOPO fold (patient B) =="
python train_tcgn.py --protocol lopo --fold 1 --epochs 2 --tag tcgn_timing

echo "== 4. epoch probe: fold B, high ceiling, snapshot every 10 =="
python train_tcgn.py --protocol lopo --fold 1 --epochs 80 --save_every 10 --tag tcgn_probe_B
# -> inspect val_mse curve in tcgn_out/tcgn_probe_B/B/run.json, then set EPOCHS
#    below to where held-out MSE plateaus (declare it; do NOT pick per fold).

echo "== 5. full 8-fold LOPO at the declared budget =="
python run_lopo.py --protocol lopo --epochs "${EPOCHS}"

echo "== 6. score =="
python score.py --tag "tcgn_lopo_833_e${EPOCHS}"

echo "done. Commit results/tcgn_lopo_833_e${EPOCHS}/{headline.json,per_fold_summary.csv}."
