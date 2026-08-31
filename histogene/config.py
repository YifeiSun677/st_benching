"""Configuration for the HisToGene port.

Paths default to the st_benching repo layout and are all overridable by
environment variable, so the same code runs on the pod and on the Mac.
"""
import os
from pathlib import Path

# st_benching repo root (this file is <repo>/histogene/config.py)
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- paths -----
WORKSPACE = Path(os.environ.get("HTG_WORKSPACE", "/workspace"))

# her2st data root: the directory that CONTAINS ST-cnts / ST-imgs / ST-spotfiles
HER2ST_DIR = Path(os.environ.get("HTG_HER2ST", WORKSPACE / "her2st" / "data"))

# clone of https://github.com/maxpmx/HisToGene  (we import vis_model from here)
HISTOGENE_REPO = Path(os.environ.get("HTG_REPO", WORKSPACE / "HisToGene"))

# one-off patch cache (built by histogene/build_cache.py) - large, not in git
CACHE_DIR = Path(os.environ.get("HTG_CACHE", WORKSPACE / "cache" / "htg_patch112"))

# raw run outputs: predictions + checkpoints - large, not in git
OUT_DIR = Path(os.environ.get("HTG_OUT", WORKSPACE / "runs"))

# scored tables - small, these DO go in git, same convention as the other models
RESULTS_DIR = Path(os.environ.get("HTG_RESULTS", REPO_ROOT / "results"))

# benchmark-wide gene panel, already in the repo
PANEL_FILE = Path(os.environ.get("HTG_PANEL", REPO_ROOT / "panels" / "panel_833.txt"))

# benchmark-wide scoring gene sets (gene_set_all/hvg/svg/marker.txt), optional
GENE_SETS_DIR = Path(os.environ.get("HTG_GENE_SETS", REPO_ROOT / "results" / "gene_sets"))

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
