#!/usr/bin/env bash
# BLEEP patient-B colour ablation driver.
# Flags match bleep/train_bleep.py and bleep/infer_bleep.py as they exist.
#
#   ./bleep/run_gray_patientB.sh smoke   # 1 fold, 2 epochs, both arms
#   ./bleep/run_gray_patientB.sh arm2    # 6 folds, inference only from arm-1 ckpts
#   ./bleep/run_gray_patientB.sh arm3    # 6 folds, train 20 epochs on grayscale
#   ./bleep/run_gray_patientB.sh all
#
# Run from /workspace/st_benching. ARM1_RUN must point at the finished colour
# run -- get it from `grep OUTROOT bleep/run_patient.sh`.

set -euo pipefail

MODE="${1:-}"
[[ -z "$MODE" ]] && { echo "usage: $0 {smoke|arm2|arm3|all}" >&2; exit 1; }

REPO="${REPO:-/workspace/st_benching}"
ROOT="${ROOT:-/workspace/her2st/data}"          # --root
CACHE="${CACHE:-/workspace/her2st_cache}"       # --cache
PANEL="${PANEL:-$REPO/panels/panel_833.txt}"    # --panel

ARM1_RUN="${ARM1_RUN:?set ARM1_RUN to the finished colour run dir}"
ARM2_RUN="${ARM2_RUN:-$(dirname "$ARM1_RUN")/bleep_patientB_833_graytest}"
ARM3_RUN="${ARM3_RUN:-$(dirname "$ARM1_RUN")/bleep_patientB_833_graytrain}"

PATIENT="${PATIENT:-B}"
EPOCHS="${EPOCHS:-20}"          # locked to the control
BATCH_SIZE="${BATCH_SIZE:-256}" # train_bleep default
INFER_BATCH="${INFER_BATCH:-128}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-50}"
METHOD="${METHOD:-average}"

SECTIONS=(B1 B2 B3 B4 B5 B6)

# The ported modules use flat imports (from her2st_dataset import ...), so
# they are run as scripts from inside bleep/, exactly as run_patient.sh does.
cd "$REPO/bleep"
log() { echo -e "\n=== $* ===\n"; }

# The checkpoint filename is whatever train_bleep.py writes; find it rather
# than assume it.
find_ckpt() {  # $1 = directory to search
  find "$1" \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) -size +1M \
    | sort | tail -n 1
}

if [[ "$MODE" == "smoke" ]]; then
  SECTIONS=(B1); EPOCHS=2
  ARM2_RUN="${ARM2_RUN}_smoke"; ARM3_RUN="${ARM3_RUN}_smoke"
  log "SMOKE TEST: fold B1 only, ${EPOCHS} epochs. Results are throwaway."
fi

# ---------------------------------------------------------------- arm 2 ----
run_arm2() {
  log "ARM 2  colour train -> grayscale test   ($ARM2_RUN)"
  for i in "${!SECTIONS[@]}"; do
    SEC="${SECTIONS[$i]}"
    FOLD=$(printf "fold%02d_%s" "$i" "$SEC")
    CKPT=$(find "$ARM1_RUN" -path "*${SEC}*" \
        \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \) -size +1M \
        | sort | tail -n 1)
    [[ -z "$CKPT" ]] && { echo "FATAL: no arm-1 checkpoint for $SEC" >&2; exit 2; }
    echo "[$FOLD] ckpt: $CKPT"
    python infer_bleep.py \
      --root "$ROOT" --panel "$PANEL" --cache "$CACHE" \
      --ckpt "$CKPT" --out "$ARM2_RUN/$FOLD" \
      --patient "$PATIENT" --test_section "$SEC" \
      --top_k "$TOP_K" --method "$METHOD" \
      --batch_size "$INFER_BATCH" --num_workers "$WORKERS" \
      --gray query
  done
}

# ---------------------------------------------------------------- arm 3 ----
run_arm3() {
  log "ARM 3  grayscale train -> grayscale test   ($ARM3_RUN)"
  for i in "${!SECTIONS[@]}"; do
    SEC="${SECTIONS[$i]}"
    FOLD=$(printf "fold%02d_%s" "$i" "$SEC")
    echo "[$FOLD] held out: $SEC"
    python train_bleep.py \
      --root "$ROOT" --panel "$PANEL" --cache "$CACHE" \
      --out "$ARM3_RUN/$FOLD" --fold_name "$FOLD" \
      --patient "$PATIENT" --test_section "$SEC" \
      --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
      --num_workers "$WORKERS" --seed "$SEED" \
      --gray all

    CKPT=$(find_ckpt "$ARM3_RUN/$FOLD")
    [[ -z "$CKPT" ]] && { echo "FATAL: train wrote no checkpoint in $ARM3_RUN/$FOLD" >&2; exit 2; }
    python infer_bleep.py \
      --root "$ROOT" --panel "$PANEL" --cache "$CACHE" \
      --ckpt "$CKPT" --out "$ARM3_RUN/$FOLD" \
      --patient "$PATIENT" --test_section "$SEC" \
      --top_k "$TOP_K" --method "$METHOD" \
      --batch_size "$INFER_BATCH" --num_workers "$WORKERS" \
      --gray all
  done
}

case "$MODE" in
  smoke) run_arm2; run_arm3 ;;
  arm2)  run_arm2 ;;
  arm3)  run_arm3 ;;
  all)   run_arm2; run_arm3 ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

log "done. arm2=$ARM2_RUN  arm3=$ARM3_RUN"
cat <<EOF
next:
  cd "$REPO" && python -m bleep.score_gray_paired \\
    --arm1 "$ARM1_RUN" --arm2 "$ARM2_RUN" --arm3 "$ARM3_RUN" \\
    --out  "$REPO/results/bleep_gray_patientB"
EOF
