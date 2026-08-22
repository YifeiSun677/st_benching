#!/usr/bin/env bash
# End-to-end DeepPT LOPO on her2st, 833-gene panel.
# Run 00_setup.sh first, then read INSPECT.md, THEN this.
set -euo pipefail

WORK=${WORK:-/workspace}
HER2ST="$WORK/her2st"
PANEL="$WORK/st_benching/panels/panel_833.txt"
WEIGHTS="$WORK/DeepPT_original/ResNet50_IMAGENET1K_V2.pt"
TAG=${TAG:-raw}
BASE="$WORK/deeppt"

FEAT="$BASE/features_$TAG"
TARG="$BASE/targets"
RES="$BASE/results/deeppt_833_$TAG"

eval "$(conda shell.bash hook)"; conda activate deeppt
cd "$(dirname "$0")"

# --- 1. features (ONE-OFF: frozen ResNet50, does not repeat per fold) -----
python 01_extract_features.py \
    --her2st "$HER2ST" --weights "$WEIGHTS" --out "$FEAT" --colornorm "$TAG"

# --- 2. targets ----------------------------------------------------------
python 02_build_targets.py \
    --her2st "$HER2ST" --panel "$PANEL" --features "$FEAT" --out "$TARG"

# --- 3. LOPO: 8 folds, AE + predictor refit inside each ------------------
python 03_run_lopo.py \
    --features "$FEAT" --targets "$TARG" --out "$RES" --tag "$TAG" \
    2>&1 | tee "$BASE/lopo_$TAG.log"

# --- 4. score: per-fold per-gene PCC, NOT pooled -------------------------
python 04_score.py \
    --preds "$RES/preds" --targets "$TARG" --features "$FEAT" \
    --out "$RES" --epoch best

# --- 5. sync the small tables into the benchmark repo -------------------
mkdir -p "$WORK/st_benching/results/deeppt_833_$TAG"
cp "$RES"/{per_gene_pcc.csv,per_fold.csv,run_summary.csv,config.json} \
   "$WORK/st_benching/results/deeppt_833_$TAG/"
echo "commit those; leave preds/*.npz and ckpt/*.pt out of git (rsync to Mac)"
