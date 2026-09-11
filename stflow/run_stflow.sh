#!/usr/bin/env bash
# stflow_port/run_stflow.sh -- one entry point for every step of the runbook.
#   bash stflow_port/run_stflow.sh <command> [args]
# Every command tees its output to $WORKSPACE/logs/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
LOG=$WORKSPACE/logs
mkdir -p "$LOG"
PIN=880c2eeaa64aaf64ffcd30f3c3278bdab4ab1dbe
cmd=${1:-help}; shift || true

case "$cmd" in
  setup)
    pip install -q -r "$HERE/requirements_stflow.txt"
    if [ ! -d "$STFLOW_REPO/.git" ]; then
      mkdir -p "$(dirname "$STFLOW_REPO")"
      git clone https://github.com/Graph-and-Geometric-Learning/STFlow.git "$STFLOW_REPO"
    fi
    git -C "$STFLOW_REPO" checkout -q "$PIN"
    echo "STFlow clone at $(git -C "$STFLOW_REPO" rev-parse --short HEAD) (read-only: never edit it)"
    ;;
  uni)
    : "${HF_TOKEN:?export HF_TOKEN=hf_xxx first (token of the account with MahmoodLab/UNI access)}"
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download
d = os.path.dirname(os.environ["UNI_CKPT"]); os.makedirs(d, exist_ok=True)
p = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=d,
                    token=os.environ["HF_TOKEN"])
print(p, f"{os.path.getsize(p)/2**30:.2f} GB")
PY
    ;;
  preflight)
    python "$HERE/preflight.py" --stage env model data uni "$@" 2>&1 | tee "$LOG/stflow_preflight.log" ;;
  features)
    python "$HERE/build_features.py" "$@" 2>&1 | tee -a "$LOG/stflow_features.log" ;;
  check_cache)
    python "$HERE/preflight.py" --stage cache --cache_tag "$CACHE_TAG" "$@" 2>&1 | tee "$LOG/stflow_check_cache.log" ;;
  overfit)
    python "$HERE/train.py" --tag stflow_overfit --overfit_section B1 --epochs 300 \
      --dropout 0 --attn_dropout 0 --cache_tag "$CACHE_TAG" --overwrite "$@" 2>&1 | tee "$LOG/stflow_overfit.log" ;;
  timing)
    python "$HERE/train.py" --tag stflow_timing --test_patient A --epochs 3 \
      --cache_tag "$CACHE_TAG" --overwrite "$@" 2>&1 | tee "$LOG/stflow_timing.log" ;;
  probe)
    python "$HERE/train.py" --tag stflow_probe_833 --test_patient A --val_patient B --skip_test \
      --epochs 200 --eval_every 5 --cache_tag "$CACHE_TAG" "$@" 2>&1 | tee "$LOG/stflow_probe.log" ;;
  lopo)
    E=${1:?usage: run_stflow.sh lopo <epochs> [extra train.py args]}; shift
    TAG=stflow_lopo_833_${CACHE_TAG#uni_v1_}_e${E}
    for P in A B C D E F G H; do
      python "$HERE/train.py" --tag "$TAG" --test_patient "$P" --epochs "$E" \
        --cache_tag "$CACHE_TAG" "$@" 2>&1 | tee -a "$LOG/$TAG.log"
    done
    python "$HERE/score.py" --tag "$TAG" 2>&1 | tee -a "$LOG/$TAG.log" ;;
  perturb)
    SRC=${1:?usage: run_stflow.sh perturb <trained_tag> gray|shuffle}; KIND=${2:?gray|shuffle}
    if [ "$KIND" = gray ]; then EXTRA=(--cache_tag "${CACHE_TAG}_gray"); else EXTRA=(--cache_tag "$CACHE_TAG" --shuffle_test_features); fi
    for P in A B C D E F G H; do
      python "$HERE/train.py" --tag "${SRC}_${KIND}" --from_tag "$SRC" --test_patient "$P" "${EXTRA[@]}" \
        2>&1 | tee -a "$LOG/${SRC}_${KIND}.log"
    done
    python "$HERE/score.py" --tag "${SRC}_${KIND}" 2>&1 | tee -a "$LOG/${SRC}_${KIND}.log" ;;
  score)
    python "$HERE/score.py" --tag "${1:?usage: run_stflow.sh score <tag>}" "${@:2}" ;;
  *)
    echo "commands: setup | uni | preflight | features [--patch_mode ..] [--grayscale] | check_cache |"
    echo "          overfit | timing | probe | lopo <E> | perturb <tag> gray|shuffle | score <tag>" ;;
esac
