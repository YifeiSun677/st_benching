"""
Central configuration for the TRIPLEX hand-port in st_benching.

Everything you might want to change lives here. Paths default to the RunPod
layout you already use (only /workspace survives a pod rebuild), and to the
same shared her2st 224px patch cache the other ports use.

Mirrors the convention of bleep/histogene/hist2st/path2space: import the real
model from a READ-ONLY upstream clone, reproduce the dataset contract, train
with our own LOPO harness, score with our own scorer.
"""
import os

# ---------------------------------------------------------------- paths
# Read-only clone of the upstream repo (we import ONLY the model from it).
TRIPLEX_REPO   = os.environ.get("TRIPLEX_REPO", "/workspace/TRIPLEX")

# Shared her2st 224x224 patch cache (the one built for BLEEP).
#   HER2ST_CACHE : uint8 memmap, shape (N_TOTAL, 224, 224, 3)
#   The per-section feature files carry a `cache_idx` column that indexes rows
#   of this memmap, so target / global / neighbor features all come from the
#   same patches as every other model in the benchmark.
HER2ST_CACHE       = os.environ.get("HER2ST_CACHE", "/workspace/her2st_cache/patches.dat")
HER2ST_CACHE_SHAPE = (int(os.environ.get("HER2ST_N", "13620")), 224, 224, 3)

# Spot index: one row per spot, columns:
#   idx, section, patient, spot_id, array_row, array_col, cache_idx
# (idx == row in the global cache; cache_idx == same thing, kept explicit).
# You almost certainly already have an equivalent from the BLEEP cache build;
# her2st.build_spot_index() will (re)build it from raw her2st if you don't.
SPOT_INDEX = os.environ.get("SPOT_INDEX", "/workspace/her2st_cache/spot_index.csv")

# Raw her2st (only needed if you have to (re)build the cache / index / counts).
HER2ST_ROOT = os.environ.get("HER2ST_ROOT", "/workspace/her2st/data")

# 833-gene panel and gene-set lists (already in the repo).
PANEL_833  = os.environ.get("PANEL_833", "/workspace/st_benching/panels/panel_833.txt")
GENE_SETS  = os.environ.get("GENE_SETS", "/workspace/st_benching/results/gene_sets")

# Where features and outputs go. Features are re-derivable, so keep them on the
# volume but out of git; outputs (small CSV/JSON/npz) get rsynced to the Mac.
FEATURE_DIR = os.environ.get("TRIPLEX_FEATURES", "/workspace/triplex_features")
OUTPUT_DIR  = os.environ.get("TRIPLEX_OUT",      "/workspace/triplex_results")

# CIGAR ResNet18 self-supervised weights (target + context share this encoder).
# The upstream model hard-codes ./weights/cigar/tenpercent_resnet18.ckpt, so we
# keep one canonical copy and symlink ./weights -> here (see run.sh).
CIGAR_CKPT = os.environ.get(
    "CIGAR_CKPT", "/workspace/weights/cigar/tenpercent_resnet18.ckpt")
CIGAR_URL  = ("https://github.com/ozanciga/self-supervised-histopathology/"
              "releases/download/tenpercent/tenpercent_resnet18.ckpt")

# ---------------------------------------------------------------- model / data
EMB_DIM      = 512          # CIGAR ResNet18 feature width (fixed)
NUM_GENES    = 833          # our common panel
RES_NEIGHBOR = (5, 5)       # 5x5 neighbour grid -> 25 tokens, centre = index 12
POS_LAYER    = "APEG"       # 'APEG' | 'MLP' | 'None' (APEG needs depth2 > 1)

# TRIPLEX andersson model config (verbatim from config/ST/andersson/TRIPLEX.yaml)
MODEL_KWARGS = dict(
    num_genes=NUM_GENES, emb_dim=EMB_DIM,
    depth1=1, depth2=3, depth3=3,
    num_heads1=4, num_heads2=16, num_heads3=16,
    mlp_ratio1=4, mlp_ratio2=4, mlp_ratio3=1,
    dropout1=0.2, dropout2=0.1, dropout3=0.3,
    kernel_size=3, res_neighbor=RES_NEIGHBOR, pos_layer=POS_LAYER,
)

# Target transform. TRIPLEX's andersson default is cpm + log1p (+ optional
# neighbourhood smoothing). We default SMOOTH=False so the headline number is
# cross-model comparable with BLEEP/Path2Space (CPM -> natural log1p); the
# smoothed variant is reported separately, exactly like Path2Space's KDTree run.
CPM    = True
SMOOTH = False

# ---------------------------------------------------------------- training
# Declared UP FRONT, scored at the LAST epoch (no selection on held-out data),
# matching the benchmark-wide rule. Time a probe first (see runbook) then fix it.
EPOCHS        = int(os.environ.get("TRIPLEX_EPOCHS", "200"))
LR            = 1.0e-4
BATCH_SIZE    = int(os.environ.get("TRIPLEX_BATCH", "128"))
NUM_WORKERS   = int(os.environ.get("TRIPLEX_WORKERS", "8"))
WEIGHT_DECAY  = 0.0
SEED          = 2021
FREEZE_TARGET = False       # set True to freeze the ResNet18 target encoder (faster, off-paper)

# her2st patients in fold order (LOPO = leave-one-patient-out, 8 folds).
PATIENTS = ["A", "B", "C", "D", "E", "F", "G", "H"]

MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]
