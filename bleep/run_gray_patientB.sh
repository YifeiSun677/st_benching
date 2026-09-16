#!/usr/bin/env bash
# BLEEP patient-B colour ablation driver.
#
#   ./bleep/run_gray_patientB.sh smoke   # 1 fold, 2 epochs, both arms
#   ./bleep/run_gray_patientB.sh arm2    # 6 folds, inference only from arm-1 ckpts
#   ./bleep/run_gray_patientB.sh arm3    # 6 folds, train 20 epochs on grayscale
#   ./bleep/run_gray_patientB.sh all     # arm2 then arm3
#
# Every knob is an env var, so nothing here needs editing to move pods:
#   CACHE=/workspace/her2st_cache RUNROOT=/workspace/runs ./bleep/run_gray_patientB.sh all

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "usage: $0 {smoke|arm2|arm3|all}" >&2; exit 1
fi

REPO="${REPO:-/workspace/st_benching}"
CACHE="${CACHE:-/workspace/her2st_cache}"
DATA="${DATA:-/workspace/her2st/data}"
PANEL="${PANEL:-$REPO/panels/panel_833.txt}"
RUNROOT="${RUNROOT:-/workspace/runs}"

ARM1_RUN="${ARM1_RUN:-$RUNROOT/bleep_patientB_833}"          # finished colour run
ARM2_RUN="${ARM2_RUN:-$RUNROOT/bleep_patientB_833_graytest}"
ARM3_RUN="${ARM3_RUN:-$RUNROOT/bleep_patientB_833_graytrain}"

EPOCHS="${EPOCHS:-20}"        # locked to the control. Do not "tune" this.
SECTIONS=(B1 B2 B3 B4 B5 B6)

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

log() { echo -e "\n=== $* ===\n"; }

# Pull the exact hyperparameters out of the finished colour run rather than
# retyping them, so arm 3 cannot silently drift from the control.
read_arm1_cfg() {
  python - "$ARM1_RUN" <<'PY'
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
js = sorted(run.rglob("run.json"))
if not js:
    sys.exit(f"no run.json under {run}")
cfg = json.loads(js[0].read_text())
flat = cfg.get("config", cfg)
for k in ("lr", "batch_size", "temperature", "top_k", "weight_decay", "seed",
          "epochs", "image_encoder", "projection_dim"):
    if k in flat:
        print(f"  {k} = {flat[k]}")
PY
}

log "arm-1 configuration on record (copy any mismatch into the calls below)"
read_arm1_cfg || true

if [[ "$MODE" == "smoke" ]]; then
  SECTIONS=(B1)
  EPOCHS=2
  ARM2_RUN="${ARM2_RUN}_smoke"
  ARM3_RUN="${ARM3_RUN}_smoke"
  log "SMOKE TEST: 1 fold (B1), ${EPOCHS} epochs. Results are throwaway."
fi

# ---------------------------------------------------------------- arm 2 ----
# Colour-trained model, grayscale query images. No training: reuse the arm-1
# checkpoints. This is the cheap arm -- run it first, it will catch a wiring
# problem in minutes instead of half an hour.
run_arm2() {
  log "ARM 2  colour train -> grayscale test   ($ARM2_RUN)"
  for i in "${!SECTIONS[@]}"; do
    SEC="${SECTIONS[$i]}"
    FOLD=$(printf "fold%02d_%s" "$i" "$SEC")
    CKPT=$(find "$ARM1_RUN" -path "*${SEC}*" \( -name "*.pt" -o -name "*.pth" \) \
           | head -n 1)
    if [[ -z "$CKPT" ]]; then
      echo "FATAL: no arm-1 checkpoint found for held-out section $SEC" >&2
      exit 2
    fi
    echo "[$FOLD] checkpoint: $CKPT"
    python -m bleep.infer \
      --data "$DATA" --cache "$CACHE" --panel "$PANEL" \
      --patient B --test_section "$SEC" \
      --checkpoint "$CKPT" \
      --gray query \
      --out "$ARM2_RUN/$FOLD"
  done
}

# ---------------------------------------------------------------- arm 3 ----
# Grayscale everywhere, training included. Same 20-epoch budget as the control.
run_arm3() {
  log "ARM 3  grayscale train -> grayscale test   ($ARM3_RUN)"
  for i in "${!SECTIONS[@]}"; do
    SEC="${SECTIONS[$i]}"
    FOLD=$(printf "fold%02d_%s" "$i" "$SEC")
    echo "[$FOLD] held out: $SEC"
    python -m bleep.train \
      --data "$DATA" --cache "$CACHE" --panel "$PANEL" \
      --patient B --test_section "$SEC" \
      --epochs "$EPOCHS" \
      --gray all \
      --out "$ARM3_RUN/$FOLD"
    python -m bleep.infer \
      --data "$DATA" --cache "$CACHE" --panel "$PANEL" \
      --patient B --test_section "$SEC" \
      --checkpoint "$ARM3_RUN/$FOLD/model_last.pt" \
      --gray all \
      --out "$ARM3_RUN/$FOLD"
  done
}

case "$MODE" in
  smoke) run_arm2; run_arm3 ;;
  arm2)  run_arm2 ;;
  arm3)  run_arm3 ;;
  all)   run_arm2; run_arm3 ;;
  *)     echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

log "done. arm2=$ARM2_RUN  arm3=$ARM3_RUN"
echo "next:"
echo "  python -m bleep.score_gray_paired \\"
echo "    --arm1 $ARM1_RUN --arm2 $ARM2_RUN --arm3 $ARM3_RUN \\"
echo "    --out $REPO/results/bleep_gray_patientB"
