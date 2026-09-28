#!/usr/bin/env bash
# Driver for the STFlow normalised-target run.
#
#   bash stflow_port/run_stflow_norm.sh preflight
#   bash stflow_port/run_stflow_norm.sh overfit
#   bash stflow_port/run_stflow_norm.sh probe
#   bash stflow_port/run_stflow_norm.sh lopo   90
#   bash stflow_port/run_stflow_norm.sh score  90
#
# Set TRAIN_ENTRY and the flag names in the block below to match your existing
# run_stflow.sh, then never touch them again. Everything downstream is frozen.

set -euo pipefail

# ---------------------------------------------------------------- edit me once
REPO="${REPO:-/workspace/st_benching}"
RUNS="${RUNS:-/workspace/runs}"
PY="${PY:-python}"

TRAIN_ENTRY="${TRAIN_ENTRY:-stflow_port/train_stflow.py}"   # <-- check this name
PANEL="${PANEL:-panels/panel_833.txt}"
FEAT_CACHE="${FEAT_CACHE:-/workspace/cache/uni_her2st}"      # <-- unchanged, reused
UNI_WEIGHTS="${UNI_WEIGHTS:-/workspace/weights/uni/pytorch_model.bin}"
OLD_RUN="${OLD_RUN:-$RUNS/stflow_lopo_833_hest112_e90}"
SEED="${SEED:-0}"                                            # <-- same as old run

# flags that are FROZEN for every stage below
COMMON="--panel $PANEL --feat_cache $FEAT_CACHE --uni_weights $UNI_WEIGHTS \
        --seed $SEED --prior_on_gpu --no_tta \
        --target_mode panel_cp10k_log1p --prior gaussian --prior_sd_floor 1e-3"
# ------------------------------------------------------------------------------

STEP="${1:-}"
ARG="${2:-}"
cd "$REPO"
mkdir -p "$RUNS"
ts() { date +%Y-%m-%dT%H:%M:%S; }

case "$STEP" in

preflight)
  $PY stflow_port/preflight_norm.py \
      --old_run "$OLD_RUN" \
      --panel "$PANEL" \
      --out "$RUNS/preflight_normtarget"
  ;;

overfit)
  # Train on B1 and predict B1. Dropout 0. Wiring check only, never reported.
  TAG="stflow_overfit_B1_normtarget"
  $PY $TRAIN_ENTRY $COMMON \
      --mode overfit --train_sections B1 --test_sections B1 \
      --epochs 300 --dropout 0.0 \
      --run_dir "$RUNS/$TAG" 2>&1 | tee "$RUNS/${TAG}.log"
  $PY stflow_port/score_norm.py --run "$RUNS/$TAG" --renorm none \
      --out "$RUNS/$TAG/scored"
  ;;

probe)
  # Same design as the old run's probe: test patient A untouched, B is the
  # validation patient, C-H train. 200 epochs, score every 10.
  TAG="stflow_probe_833_normtarget_e200"
  $PY $TRAIN_ENTRY $COMMON \
      --mode probe --train_patients C,D,E,F,G,H --val_patients B \
      --epochs 200 --eval_every 10 \
      --run_dir "$RUNS/$TAG" 2>&1 | tee "$RUNS/${TAG}.log"
  echo "$(ts) probe done. Read the val curve and DECLARE the epoch budget now."
  ;;

lopo)
  [ -n "$ARG" ] || { echo "usage: $0 lopo <EPOCHS>"; exit 1; }
  TAG="stflow_lopo_833_normtarget_e${ARG}"
  for P in A B C D E F G H; do
    echo "$(ts) fold $P"
    $PY $TRAIN_ENTRY $COMMON \
        --mode lopo --test_patient "$P" --epochs "$ARG" \
        --run_dir "$RUNS/$TAG" 2>&1 | tee -a "$RUNS/${TAG}.log"
  done
  ;;

score)
  [ -n "$ARG" ] || { echo "usage: $0 score <EPOCHS>"; exit 1; }
  NEW="$RUNS/stflow_lopo_833_normtarget_e${ARG}"
  $PY stflow_port/score_norm.py --run "$NEW" --renorm none --out "$NEW/scored"
  echo "--- old run rescored on the same footing ---"
  $PY stflow_port/score_norm.py --run "$OLD_RUN" --renorm panel_cp10k \
      --out "$OLD_RUN/scored_cp10k"
  ;;

*)
  echo "usage: $0 {preflight|overfit|probe|lopo <E>|score <E>}"; exit 1 ;;
esac
