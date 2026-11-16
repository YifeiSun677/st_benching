import os
from pathlib import Path

HER2ST_DIR = Path(os.environ.get("HER2ST_DIR", "/workspace/her2st/data"))
CNT_DIR = HER2ST_DIR / "ST-cnts"
IMG_DIR = HER2ST_DIR / "ST-imgs"
POS_DIR = HER2ST_DIR / "ST-spotfiles"

REPO_ROOT   = Path(__file__).resolve().parents[1]
PANEL_FILE  = Path(os.environ.get("PANEL_FILE", REPO_ROOT / "panels/panel_833.txt"))
CACHE_DIR   = Path(os.environ.get("HIST2ST_CACHE", "/workspace/her2st_cache_112"))
OUT_ROOT    = Path(os.environ.get("HIST2ST_OUT", "/workspace/runs"))
HIST2ST_REPO = Path(os.environ.get("HIST2ST_REPO", "/workspace/repos/Hist2ST"))

PATCH = 112          # Hist2ST fig_size = 112 (r = 224//4)
R = PATCH // 2       # 56
PATIENTS = list("ABCDEFGH")

# target normalisation:
#   "cp10k"  = per-row scale to 10000 + log10(x+1)  <- Hist2ST/HisToGene original:
#              scprep.normalize.library_size_normalize defaults to rescale=10000
#   "median" = per-row scale to the section's median library size + log10(x+1).
#              On the sparse 833 panel this gives targets with variance ~0.02, so the MSE
#              term is swamped by zinb*ZINB and the target scale differs between sections.
# MUST match whatever your histogene port uses, otherwise MSE / SSE ratio are not
# comparable across the two models (PCC is unaffected).
NORM = os.environ.get("HIST2ST_NORM", "cp10k")
