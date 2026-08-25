#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DeepPT on her2st — environment setup + original repo download
# Run once per RunPod pod. Everything lands on the network volume (/workspace).
# ---------------------------------------------------------------------------
set -euo pipefail

WORK=${WORK:-/workspace}
DEEPPT_DIR="$WORK/DeepPT_original"     # the unmodified Zenodo release
CODE_DIR="$WORK/st_benching"           # your benchmark repo
HER2ST="$WORK/her2st"                  # your her2st copy

# --- 1. conda env --------------------------------------------------------
# NOTE: do NOT pin torch==1.12.1 on a 4090 (sm_89 needs CUDA 11.8+/12.x).
# NOTE: openslide is NOT installed — her2st ships JPEG, not .svs.

pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib pillow tqdm
pip install torchstain          # only needed if you run --colornorm macenko

python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("torchvision", torchvision.__version__)
PY

# --- 2. the original DeepPT release (Zenodo, not GitHub) ------------------
mkdir -p "$DEEPPT_DIR" && cd "$DEEPPT_DIR"
BASE="https://zenodo.org/records/11125591/files"
for f in 10metadata.zip 11slide_processing.zip 12AE.zip 13DeepPT_train.zip README.docx; do
    [ -f "$f" ] || curl -L -o "$f" "$BASE/$f?download=1"
done
# 102.5 MB — the exact ResNet50 IMAGENET1K_V2 weights the paper used.
[ -f ResNet50_IMAGENET1K_V2.pt ] || \
    curl -L -o ResNet50_IMAGENET1K_V2.pt "$BASE/ResNet50_IMAGENET1K_V2.pt?download=1"

for z in 10metadata 11slide_processing 12AE 13DeepPT_train; do
    [ -d "$z" ] || unzip -q -o "$z.zip"
done

# Verify the checkpoint md5 (from the Zenodo record)
md5sum ResNet50_IMAGENET1K_V2.pt
echo "expected: 1160a97591960c585812c64efbd79de0"

# --- 3. your code + data -------------------------------------------------
cd "$WORK"
[ -d "$CODE_DIR" ] || git clone https://github.com/YifeiSun677/st_benching.git
[ -d "$HER2ST" ]   || git clone https://github.com/almaan/her2st.git

# her2st ships its images with git-lfs
cd "$HER2ST" && git lfs install && git lfs pull && cd "$WORK"

# --- 4. sanity ------------------------------------------------------------
ls "$HER2ST/data/ST-cnts" | head
ls "$HER2ST/data/ST-spotfiles" | head
ls "$HER2ST/data/ST-imgs"
wc -l "$CODE_DIR/panels/panel_833.txt"

echo
echo "=== NEXT: read the original code before running anything ==="
echo "  see INSPECT.md — you must transplant the real hyperparameters"
echo "  from $DEEPPT_DIR/12AE/1main_AE.py and 13DeepPT_train/1main_train.py"
