#!/usr/bin/env bash
# End-to-end TRIPLEX port on a fresh RunPod. Assumes /workspace is your
# persistent (manually created) volume and that the her2st 224px cache +
# spot index already live there. Run from the st_benching repo root.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/st_benching}"
export TRIPLEX_REPO="${TRIPLEX_REPO:-/workspace/TRIPLEX}"
export CIGAR_CKPT="${CIGAR_CKPT:-/workspace/weights/cigar/tenpercent_resnet18.ckpt}"
cd "$REPO_ROOT"

echo "== 1. read-only clone of upstream TRIPLEX (model source only) =="
if [ ! -d "$TRIPLEX_REPO/.git" ]; then
  git clone --depth 1 https://github.com/NEXGEM/TRIPLEX.git "$TRIPLEX_REPO"
fi

echo "== 2. deps (no flash-attn / HEST / CLAM) =="
pip install --no-input -r triplex/requirements-triplex.txt

echo "== 3. CIGAR ResNet18 weights (once; cached on the volume) =="
mkdir -p "$(dirname "$CIGAR_CKPT")"
if [ ! -f "$CIGAR_CKPT" ]; then
  wget -O "$CIGAR_CKPT" \
    https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt
fi
# the upstream model init looks for ./weights/cigar relative to CWD; point it here
mkdir -p weights && ln -sfn "$(dirname "$(dirname "$CIGAR_CKPT")")" weights_link_tmp 2>/dev/null || true
mkdir -p weights/cigar && ln -sfn "$CIGAR_CKPT" weights/cigar/tenpercent_resnet18.ckpt

echo "== 4. PREFLIGHT (wiring test; needs GPU, not her2st) =="
python -m triplex.preflight

echo "== 5. build features once (CIGAR global + 5x5 neighbours per section) =="
python -m triplex.build_features

echo "== 6. SMOKE TEST: one patient, 3 epochs, then score =="
python -m triplex.run_lopo --tag triplex_smoke --patients B --epochs 3

echo "== 7. FULL LOPO (8 folds) + score =="
echo "   edit TRIPLEX_EPOCHS after checking s/epoch from the smoke test"
python -m triplex.run_lopo --tag triplex_lopo_833

echo "== done. headline at $TRIPLEX_OUT/triplex_lopo_833/headline.json =="
