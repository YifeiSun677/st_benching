#!/usr/bin/env bash
# Within-patient leave-one-section-out BLEEP run.
#
#   bash run_patient.sh B /workspace/her2st/data ../panels/panel_833.txt \
#        /workspace/runs/bleep_patientB
#
# Resumable: a fold whose preds.npz already exists is skipped.
set -euo pipefail

PATIENT="${1:-B}"
ROOT="${2:-/workspace/her2st/data}"
PANEL="${3:-../panels/panel_833.txt}"
OUTROOT="${4:-/workspace/runs/bleep_patient${PATIENT}}"
EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-256}"
WORKERS="${WORKERS:-8}"

SECTIONS=$(ls "${ROOT}/ST-cnts/" | grep -E "^${PATIENT}[0-9]\.tsv\.gz$" \
           | sed 's/\.tsv\.gz//' | sort)
echo "patient ${PATIENT} sections: ${SECTIONS}"
mkdir -p "${OUTROOT}"

for SEC in ${SECTIONS}; do
  OUT="${OUTROOT}/${SEC}"
  if [ -f "${OUT}/preds.npz" ]; then
    echo "=== ${SEC}: already done, skipping ==="
    continue
  fi
  echo "=== fold ${SEC} ==="
  mkdir -p "${OUT}"

  python train_bleep.py --root "${ROOT}" --panel "${PANEL}" \
    --patient "${PATIENT}" --test_section "${SEC}" --out "${OUT}" \
    --epochs "${EPOCHS}" --batch_size "${BATCH}" --num_workers "${WORKERS}" \
    2>&1 | tee "${OUT}/train.log"

  python infer_bleep.py --root "${ROOT}" --panel "${PANEL}" \
    --patient "${PATIENT}" --test_section "${SEC}" \
    --ckpt "${OUT}/last.pt" --out "${OUT}" --num_workers "${WORKERS}" \
    2>&1 | tee "${OUT}/infer.log"

  python diagnostics.py --preds "${OUT}/preds.npz" --ckpt "${OUT}/last.pt" \
    --root "${ROOT}" --panel "${PANEL}" --patient "${PATIENT}" \
    --test_section "${SEC}" 2>&1 | tee "${OUT}/diagnostics.log"
done

echo "=== scoring ==="
python score_bleep.py --run_dir "${OUTROOT}" \
  --out_dir "../results/bleep_patient${PATIENT}_833"
echo "done"
