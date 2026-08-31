#!/usr/bin/env bash
# Leave-one-SECTION-out on the repo's own split (samples = names[1:33]).
# Default folds are one section per patient: A2 B1 C1 D1 E1 F1 G1
# (indices into samples[]: 0 5 11 17 23 26 29).
# Run from the st_benching repo root:   bash histogene/run_loso.sh [TAG] [FOLDS...]
set -euo pipefail
TAG="${1:-histogene_loso_833}"; shift || true
FOLDS=("$@"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 5 11 17 23 26 29)
LOG="${HTG_OUT:-/workspace/runs}/${TAG}/logs"
mkdir -p "$LOG"
for f in "${FOLDS[@]}"; do
  echo "===== fold $f ====="
  python -m histogene.train --cv section --fold "$f" --tag "$TAG" \
    2>&1 | tee "$LOG/fold${f}.log"
done
echo "all folds done"
