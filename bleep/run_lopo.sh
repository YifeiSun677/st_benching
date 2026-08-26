#!/usr/bin/env bash
# Leave-one-patient-out BLEEP run over all her2st patients.
#
#   bash run_lopo.sh /workspace/her2st/data ../panels/panel_833.txt \
#        /workspace/runs/bleep_lopo /workspace/her2st_cache
#
# EPOCHS controls the budget. See the note on step-matching in README:
# LOPO folds have ~7x the training spots of the patient-B control, so
# equal EPOCHS is NOT equal optimiser steps.
#
# Resumable: a fold with an existing preds.npz is skipped.
set -euo pipefail

ROOT="${1:-/workspace/her2st/data}"
PANEL="${2:-../panels/panel_833.txt}"
OUTROOT="${3:-/workspace/runs/bleep_lopo}"
CACHE="${4:-}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-8}"

CACHE_ARG=""
[ -n "${CACHE}" ] && CACHE_ARG="--cache ${CACHE}"

PATIENTS=$(python -c "
import splits; print(' '.join(splits.all_patients('${ROOT}')))")
echo "patients: ${PATIENTS}"
echo "epochs=${EPOCHS} batch=${BATCH} cache='${CACHE}'"
mkdir -p "${OUTROOT}"

for P in ${PATIENTS}; do
  OUT="${OUTROOT}/${P}"
  if [ -f "${OUT}/preds.npz" ]; then
    echo "=== ${P}: done, skipping ==="
    continue
  fi
  echo "=== held-out patient ${P} ==="
  mkdir -p "${OUT}"

  read -r TRAIN TEST < <(python -c "
import splits
tr, te = splits.lopo_split('${ROOT}', '${P}')
print(','.join(tr), ','.join(te))")
  echo "  train: ${TRAIN}"
  echo "  test : ${TEST}"

  python train_bleep.py --root "${ROOT}" --panel "${PANEL}" ${CACHE_ARG} \
    --train_sections "${TRAIN}" --test_sections "${TEST}" \
    --fold_name "${P}" --out "${OUT}" --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" --num_workers "${WORKERS}" \
    2>&1 | tee "${OUT}/train.log"

  python infer_bleep.py --root "${ROOT}" --panel "${PANEL}" ${CACHE_ARG} \
    --train_sections "${TRAIN}" --test_sections "${TEST}" \
    --ckpt "${OUT}/last.pt" --out "${OUT}" --num_workers "${WORKERS}" \
    2>&1 | tee "${OUT}/infer.log"

  python diagnostics.py --preds "${OUT}/preds.npz" --ckpt "${OUT}/last.pt" \
    --root "${ROOT}" --panel "${PANEL}" ${CACHE_ARG} \
    --test_sections "${TEST}" 2>&1 | tee "${OUT}/diagnostics.log"
done

echo "=== scoring ==="
python score_bleep.py --run_dir "${OUTROOT}" \
  --out_dir "../results/bleep_lopo_833_e${EPOCHS}"
echo done
