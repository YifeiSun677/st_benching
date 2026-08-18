#!/usr/bin/env bash
# Run once on a fresh pod, from /workspace.
#   bash setup_pod.sh /workspace/her2st
set -euo pipefail

HER2ST="${1:-/workspace/her2st}"
ROOT=/workspace/HisToGene

echo "== deps =="
# torch is already present in the RunPod image; do not reinstall it.
pip install -q "pytorch-lightning==1.9.5" "torchmetrics==0.11.4" \
               einops scprep scanpy anndata pandas pillow tqdm scikit-learn

echo "== HisToGene repo =="
[ -d "$ROOT" ] || git clone -q https://github.com/maxpmx/HisToGene.git "$ROOT"
cd "$ROOT"
mkdir -p model processed logs cache panels

echo "== her2st =="
[ -e data/her2st ] || ln -s "$HER2ST" data/her2st
ls data/her2st/data/ST-cnts | wc -l   # expect 36

echo "== benchmark code =="
# st_benching holds the shared panels + these HisToGene scripts.
if [ -d /workspace/st_benching ]; then
  git -C /workspace/st_benching pull -q
else
  git clone -q https://github.com/YifeiSun677/st_benching.git /workspace/st_benching
fi
cp /workspace/st_benching/models/histogene/*.py "$ROOT"/
cp /workspace/st_benching/panels/panel_833.txt "$ROOT"/panels/

echo
echo "ready. next:"
echo "  cd $ROOT"
echo "  python build_cache.py --her2st data/her2st --out cache"
echo "  python smoke_test.py --panel panels/panel_833.txt"
