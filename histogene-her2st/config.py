"""Central configuration.

Every path can be overridden with an environment variable so the same code runs
on the pod and on the Mac without editing files.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths -----
WORKSPACE = Path(os.environ.get("HTG_WORKSPACE", "/workspace"))

# her2st data root: the directory that CONTAINS ST-cnts / ST-imgs / ST-spotfiles
HER2ST_DIR = Path(os.environ.get("HTG_HER2ST", WORKSPACE / "her2st" / "data"))

# clone of https://github.com/maxpmx/HisToGene  (we import vis_model from here)
HISTOGENE_REPO = Path(os.environ.get("HTG_REPO", WORKSPACE / "HisToGene"))

# one-off patch cache (built by scripts/01_build_cache.py)
CACHE_DIR = Path(os.environ.get("HTG_CACHE", WORKSPACE / "cache" / "htg_patch112"))

# run outputs
OUT_DIR = Path(os.environ.get("HTG_OUT", WORKSPACE / "runs"))

# gene panel, one HGNC symbol per line
PANEL_FILE = Path(os.environ.get("HTG_PANEL", "panels/panel_833.txt"))

CNT_DIR = HER2ST_DIR / "ST-cnts"
IMG_DIR = HER2ST_DIR / "ST-imgs"
POS_DIR = HER2ST_DIR / "ST-spotfiles"

# ------------------------------------------------------ model / training ----
# ViT_HER2ST in the original repo uses r = 224 // 4 = 56, i.e. a 112x112 patch.
PATCH_R = int(os.environ.get("HTG_PATCH_R", 56))
PATCH_SIZE = 2 * PATCH_R                      # 112
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE       # 37632

N_POS = int(os.environ.get("HTG_N_POS", 64))          # nn.Embedding size for array coords
N_LAYERS = int(os.environ.get("HTG_N_LAYERS", 8))     # tutorial.ipynb uses 8
DIM = int(os.environ.get("HTG_DIM", 1024))
DROPOUT = float(os.environ.get("HTG_DROPOUT", 0.1))
LR = float(os.environ.get("HTG_LR", 1e-5))            # tutorial.ipynb default
EPOCHS = int(os.environ.get("HTG_EPOCHS", 100))       # fixed budget, scored at LAST epoch
SEED = int(os.environ.get("HTG_SEED", 0))

# cross-validation scheme: "patient" (8 folds, A-H) or "section" (repo-native)
CV = os.environ.get("HTG_CV", "patient")
