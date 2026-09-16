#!/usr/bin/env bash
# HisToGene colour ablation, LOPO, 833 panel.
# Run from the st_benching repo root, inside tmux:
#
#   bash histogene/run_gray_lopo.sh smoke     # fold 0, 2 epochs, ~2 min, throwaway
#   bash histogene/run_gray_lopo.sh timing    # fold 0, 5 epochs, measures s/epoch
#   bash histogene/run_gray_lopo.sh full      # 8 folds x 100 epochs x 2 models
#   bash histogene/run_gray_lopo.sh full 3 4  # only folds 3 and 4
#
# Resumable: a fold/arm with run.json is skipped. Add FORCE=1 to redo.
set -euo pipefail

MODE="${1:-full}"; shift || true
TAG="${TAG:-htg_gray_lopo_833}"
REF_TAG="${REF_TAG:-histogene_lopo_833}"      # original colour run, for reproduction check
WORKERS="${WORKERS:-4}"
EXTRA=()
[[ "${FORCE:-0}" == "1" ]] && EXTRA+=(--force)

case "$MODE" in
  smoke)  EPOCHS=2;   FOLDS=(0); TAG="${TAG}_smoke" ;;
  timing) EPOCHS=5;   FOLDS=(0); TAG="htg_gray_timing" ;;
  full)   EPOCHS=100; FOLDS=("$@"); [[ ${#FOLDS[@]} -eq 0 ]] && FOLDS=(0 1 2 3 4 5 6 7) ;;
  *) echo "usage: $0 {smoke|timing|full} [folds...]" >&2; exit 1 ;;
esac

OUT="${HTG_OUT:-/workspace/runs}/${TAG}"
LOG="${OUT}/logs"
mkdir -p "$LOG"
echo "[$(date '+%F %T')] mode=$MODE tag=$TAG epochs=$EPOCHS folds=${FOLDS[*]}"

T0=$(date +%s)
for f in "${FOLDS[@]}"; do
  echo "===== fold $f  [$(date '+%T')] ====="
  python -m histogene.train_gray --cv patient --fold "$f" --tag "$TAG" \
      --epochs "$EPOCHS" --workers "$WORKERS" "${EXTRA[@]}" \
      2>&1 | tee "$LOG/fold${f}.log"
done
echo "[$(date '+%F %T')] training done in $(( ($(date +%s) - T0) / 60 )) min"

if [[ "$MODE" == "timing" ]]; then
  python - "$OUT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for arm in ("arm1_colour", "arm3_graytrain"):
    for rj in sorted((root / arm).glob("*/run.json")):
        m = json.loads(rj.read_text())
        print(f"{arm:15s} {m['sec_per_epoch']:.3f} s/epoch  peak {m['peak_gpu_gb']} GB")
        spe = m["sec_per_epoch"]
        print(f"{'':15s} -> one 100-epoch model {spe*100/60:.1f} min")
PY
  echo "LOPO estimate = 8 x (colour model + gray model + 3 predictions)"
  echo "clean up:  rm -rf $OUT"
  exit 0
fi

REF_ARGS=()
[[ -d "${HTG_OUT:-/workspace/runs}/${REF_TAG}" && "$MODE" == "full" ]] && REF_ARGS=(--ref_tag "$REF_TAG")
python -m histogene.score_gray --tag "$TAG" "${REF_ARGS[@]}" 2>&1 | tee "$LOG/score.log"
echo "[$(date '+%F %T')] all done -> results/${TAG}/"
