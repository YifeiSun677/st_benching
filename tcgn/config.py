"""
Central config for the TCGN hand-port inside st_benching.

Everything a run touches lives here so a rebuild on a fresh RunPod is one file to
check. Paths are env-overridable; the defaults assume the RunPod layout
(/workspace is the persistent network volume).

TCGN paper/repo: Xiao et al., "Transformer with Convolution and Graph-Node
co-embedding", Medical Image Analysis 2023. Code: github.com/lugia-xiao/TCGN
(cloned READ-ONLY; we import the model unchanged and never edit the clone).
"""
import os

# --- where things are -------------------------------------------------------
# Read-only clone of the upstream repo (we only import model.py + its blocks).
TCGN_REPO_DIR = os.environ.get("TCGN_REPO_DIR", "/workspace/TCGN")
# CMT-Tiny ImageNet weights (upstream loads these strict=False before training).
CMT_WEIGHTS = os.environ.get("CMT_WEIGHTS", os.path.join(TCGN_REPO_DIR, "pretrained", "cmt_tiny.pth"))

# her2st data (the almaan/her2st clone). Same copy the other ports use.
HER2ST_DIR = os.environ.get("HER2ST_DIR", "/workspace/her2st/data")
CNT_DIR = os.path.join(HER2ST_DIR, "ST-cnts")
IMG_DIR = os.path.join(HER2ST_DIR, "ST-imgs")
POS_DIR = os.path.join(HER2ST_DIR, "ST-spotfiles")

# Common 833-gene panel (785-HVG u PAM50), kept in the st_benching repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_FILE = os.environ.get("PANEL_833", os.path.join(_HERE, "..", "panels", "panel_833.txt"))

# Patch cache (uint8 memmap, built once). Big -> put on the volume, not in git.
CACHE_DIR = os.environ.get("TCGN_CACHE", "/workspace/tcgn_cache")

# Where per-fold predictions + run.json go. Big npz stays off git (rsync to Mac);
# only the scored CSV/JSON summaries get committed under results/.
OUT_DIR = os.environ.get("TCGN_OUT", "/workspace/tcgn_out")

# --- patch geometry (verbatim from ViT_HER2ST in the TCGN repo) -------------
R = 56                 # self.r = 224 // 4 ; crop is 2*R = 112 px, native 20x
RESIZE = 224           # TCGN resizes the 112 crop up to 224 before the network
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # timm IMAGENET_DEFAULT_MEAN
IMAGENET_STD = (0.229, 0.224, 0.225)    # timm IMAGENET_DEFAULT_STD

# --- target transform -------------------------------------------------------
# TCGN's dataset does scp.transform.log(scp.normalize.library_size_normalize(X)).
# scprep defaults: library_size_normalize(rescale=10000) then log(base=10,psc=1).
#   -> target = log10( X / rowsum * 10000 + 1 ) = log10(CP10K + 1)
# This is the SAME scale as the HisToGene port, so per-gene numbers are directly
# comparable with HisToGene and (via PCC, scale-invariant) with every model.
# Set TARGET_RESCALE="median" to instead match an older-scprep median-libsize
# Hist2ST run. Normalisation is done over the 833-panel columns (subset first,
# then normalise), exactly as upstream subsets to its gene_set then normalises.
TARGET_RESCALE = os.environ.get("TCGN_RESCALE", "10000")   # "10000" (CP10K) or "median"
TARGET_LOG_BASE = 10
TARGET_PSEUDOCOUNT = 1

# --- training (from the repo's train.py, not model.py's __main__) -----------
LR = 1e-5              # single Adam group over all params (train.py), betas .9/.999
BATCH = 32
WEIGHT_DECAY = 0.0
# Fixed, pre-declared budget scored at the LAST epoch (benchmark rule). Upstream
# runs 79 epochs and picks the best epoch on the test section; we do not select
# on held-out data. Confirm/adjust with the probe (see RUNBOOK step 6).
EPOCHS = int(os.environ.get("TCGN_EPOCHS", "50"))
NUM_WORKERS = int(os.environ.get("TCGN_WORKERS", "8"))
SEED = int(os.environ.get("TCGN_SEED", "6"))   # repo uses torch.manual_seed(6)

# Marker genes reported alongside the headline (HER2 amplicon + hormone recept.)
MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]
