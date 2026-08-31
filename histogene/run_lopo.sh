#!/usr/bin/env bash
# Leave-one-PATIENT-out, 8 folds, 833-gene panel, 100 epochs each.
# Run from the st_benching repo root:   bash histogene/run_lopo.sh [TAG]
set -euo pipefail
TAG="${1:-histogene_lopo_833}"
LOG="${HTG_OUT:-/workspace/runs}/${TAG}/logs"
mkdir -p "$LOG"
for f in 0 1 2 3 4 5 6 7; do
  echo "===== fold $f ====="
  python -m histogene.train --cv patient --fold "$f" --tag "$TAG" \
    2>&1 | tee "$LOG/fold${f}.log"
done
echo "all folds done"
