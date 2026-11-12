"""
ridge/config.py -- single source of truth for the linear baselines on her2st.

Baselines implemented (all leave-one-patient-out over patients A-H):
  train_mean   predict each gene's mean over the 7 training patients
  region_mean  predict the mean of the spot's pathologist label over the training
               patients' annotated sections (scored on the 8 annotated sections only)
  ridge        closed-form ridge on FROZEN image features, alpha chosen by an inner
               leave-one-patient-out inside the 7 training patients (the held-out
               patient is never touched while choosing alpha)

Every path can be overridden by an environment variable, so nothing needs editing
on a fresh pod. Defaults follow the RunPod layout used by the other ports.
"""
import os

WS = os.environ.get("WORKSPACE", "/workspace")
ST_BENCH = os.environ.get("ST_BENCH", f"{WS}/st_benching")

# ---------------------------------------------------------------- inputs
HER2ST_ROOT = os.environ.get("HER2ST_ROOT", f"{WS}/her2st/data")      # ST-cnts/, ST-pat/lbl/
PANEL = os.environ.get("PANEL", f"{ST_BENCH}/panels/panel_833.txt")
GENE_SETS = os.environ.get("GENE_SETS", f"{ST_BENCH}/results/gene_sets")
# ST-Net's per-spot denominator (one number per spot, keyed "SEC:XxY"), recovered from
# ST-Net's stored truth by ridge.stnet_denominator; needed only for target stnet_lognorm.
STNET_DENOM = os.environ.get("STNET_DENOM", f"{ST_BENCH}/panels/stnet_denominator.csv")

# Shared 224 px cache (built for BLEEP; used by ST-Net/BLEEP/TRIPLEX). Its index.json
# fixes the canonical spot order; expression.npy is the shared CPM+log1p truth.
HER2ST_CACHE_DIR = os.environ.get("HER2ST_CACHE_DIR", f"{WS}/her2st_cache")
INDEX_JSON = os.environ.get("BLEEP_INDEX_JSON", f"{HER2ST_CACHE_DIR}/index.json")
EXPRESSION_NPY = os.environ.get("HER2ST_EXPRESSION", f"{HER2ST_CACHE_DIR}/expression.npy")
PATCHES_NPY = os.environ.get("HER2ST_CACHE", f"{HER2ST_CACHE_DIR}/patches.npy")

# Frozen-feature caches written by the other ports (re-used, never rewritten).
STFLOW_CACHE = os.environ.get("STFLOW_CACHE", f"{WS}/stflow_cache")
TRIPLEX_FEATURES = os.environ.get("TRIPLEX_FEATURES", f"{WS}/triplex_features")
DEEPPT_FEATURES = os.environ.get("DEEPPT_FEATURES", f"{WS}/deeppt/features_raw")
P2S_FEATURES = os.environ.get("P2S_FEATURES", f"{WS}/p2s_feature_cache")
UNI_CKPT = os.environ.get("UNI_CKPT", f"{WS}/weights/uni/pytorch_model.bin")

# ---------------------------------------------------------------- outputs
RIDGE_CACHE = os.environ.get("RIDGE_CACHE", f"{WS}/ridge_cache")     # aligned features, counts
OUTPUT_DIR = os.environ.get("RIDGE_OUT", f"{WS}/ridge_results")      # preds + scores

# ---------------------------------------------------------------- design
PATIENTS = list("ABCDEFGH")
EXPECTED_N_SECTIONS = 36
EXPECTED_N_SPOTS = 13620
EXPECTED_N_GENES = 833

# Feature sources. Each one is EXACTLY the frozen representation a deep model saw,
# so "same features + ridge" vs "the deep model" isolates the architecture.
#   twin_of : the deep model this source is the frozen twin of (None = general baseline)
#   target  : that model's native target space (see data.TARGETS) for the paired run
FEATURE_SOURCES = {
    "uni_hest112": dict(
        desc="UNI v1 CLS, 112 um crop resized to 224 px (UNI's 20x scale)",
        twin_of="STFlow", target="raw_ln", dim=1024),
    "uni_native224": dict(
        desc="UNI v1 CLS, native 224 px crop (same field of view as the shared cache)",
        twin_of=None, target="cpm_panel_ln", dim=1024),
    "cigar": dict(
        desc="CIGAR ResNet18 (self-supervised histology), shared 224 px cache",
        twin_of="TRIPLEX", target="cpm_panel_ln", dim=512),
    "resnet50": dict(
        desc="ImageNet ResNet50 pooled, DeepPT crops, no colour normalisation",
        twin_of="DeepPT", target="cp10k_full_log10", dim=2048),
    "ctranspath": dict(
        desc="CTransPath, Macenko-normalised 224 px crops",
        twin_of="Path2Space", target="mediannorm_panel_ln", dim=768),
}

# Ridge
ALPHAS = [10.0 ** k for k in range(-2, 9)] + [3.0 * 10.0 ** k for k in range(-2, 8)]
ALPHAS = sorted(ALPHAS)                 # 10^-2 ... 10^8, two points per decade
# (was 10^-2 ... 10^6; ctranspath/full fold E chose 10^6 = the top edge, so the grid
#  was extended. Weak features need alpha near n_train ~ 1e4 x feature scale.)
GLOBAL_PCS = 5                          # section-mean features -> PCA(5) on training sections
CLIP_AT_ZERO = False                    # pure linear output by default (no post-processing)

# Context blocks available to --context (ridge only)
CONTEXTS = {
    "local": ["local"],
    "ring1": ["local", "ring1"],
    "rings": ["local", "ring1", "ring2"],
    "full":  ["local", "ring1", "ring2", "global"],
}

MARKERS = ["ERBB2", "GRB7", "FASN", "GNAS", "ESR1", "PGR", "MKI67"]
