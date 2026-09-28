#!/usr/bin/env bash
# Full 8-fold LOPO. Run from /workspace/HisToGene after build_cache + smoke_test.
#   bash run_all_folds.sh              # 4 folds at a time on one GPU
#   PAR=1 bash run_all_folds.sh        # serial, if you hit OOM
set -euo pipefail

PANEL="${PANEL:-panels/panel_833.txt}"
TAG="${TAG:-htg_her2st_833_lopo}"
EPOCHS="${EPOCHS:-100}"
PAR="${PAR:-4}"

mkdir -p logs processed model

echo "== training: tag=$TAG panel=$PANEL epochs=$EPOCHS parallel=$PAR =="
for f in 0 1 2 3 4 5 6 7; do
  python train_lopo.py --fold "$f" --panel "$PANEL" --tag "$TAG" \
      --epochs "$EPOCHS" > "logs/train_${TAG}_${f}.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
done
wait

echo "== inference =="
for f in 0 1 2 3 4 5 6 7; do
  python predict_lopo.py --fold "$f" --panel "$PANEL" --tag "$TAG" \
      > "logs/pred_${TAG}_${f}.log" 2>&1
done

echo "== summary =="
python - <<EOF
import glob, json
for p in sorted(glob.glob("logs/${TAG}_*_run.json")):
    m = json.load(open(p))
    print(f"fold {m['fold']} ({m['held_out']}): {m['minutes']:>6.1f} min, "
          f"peak GPU {m['peak_gpu_gb']} GB")
EOF
ls -lh processed/${TAG}_fold*.npz
