"""
stflow_port/config.py -- single source of truth for paths and hyperparameters.

Every path can be overridden by an environment variable (see env.sh), so nothing
here needs editing on a new pod as long as env.sh is sourced.

Hyperparameters in UPSTREAM are copied verbatim from the argparse defaults of
Graph-and-Geometric-Learning/STFlow @ 880c2ee, stflow/app/flow/train.py, plus the
README's recommended command (--batch_size 2 --n_layers 4 --n_sample_steps 5).
"""
import os

WS = os.environ.get("WORKSPACE", "/workspace")

ST_BENCH = os.environ.get("ST_BENCH", f"{WS}/st_benching")
STFLOW_REPO = os.environ.get("STFLOW_REPO", f"{WS}/repos/STFlow")
STFLOW_PINNED_COMMIT = "880c2eeaa64aaf64ffcd30f3c3278bdab4ab1dbe"

# her2st raw data root: must contain ST-cnts/, ST-imgs/, ST-spotfiles/
HER2ST_ROOT = os.environ.get("HER2ST_ROOT", f"{WS}/her2st/data")
PANEL = os.environ.get("PANEL", f"{ST_BENCH}/panels/panel_833.txt")

UNI_CKPT = os.environ.get("UNI_CKPT", f"{WS}/weights/uni/pytorch_model.bin")
CACHE_ROOT = os.environ.get("STFLOW_CACHE", f"{WS}/stflow_cache")
RUNS_ROOT = os.environ.get("RUNS_ROOT", f"{WS}/runs")

PATIENTS = list("ABCDEFGH")
EXPECTED_N_SECTIONS = 36
EXPECTED_N_SPOTS = 13620          # the benchmark-wide her2st spot count
EXPECTED_N_GENES = 833

# ---- geometry ---------------------------------------------------------------
SPOT_PITCH_UM = 200.0             # ST v1 array: 100 um spots, 200 um centre-to-centre
HEST_PATCH_UM = 112.0             # HEST-bench patch = 224 px at 0.5 um/px (UNI's 20x scale)
UNI_INPUT_PX = 224

# ---- UNI (uni_v1_official) exactly as hest_utils/pretrained_configs/uni_v1_official.json
UNI_TIMM_KWARGS = dict(model_name="vit_large_patch16_224", dynamic_img_size=True,
                       num_classes=0, init_values=1.0)
UNI_FEATURE_DIM = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- STFlow upstream defaults (app/flow/train.py) -----------------------------
UPSTREAM = dict(
    seed=1,
    lr=5e-4,
    epochs=100,               # upstream ceiling; upstream ALSO early-stops on the test set (we do not)
    batch_size=2,
    sample_times=10,          # random sub-patches drawn per training section per epoch
    clip_norm=1.0,
    patch_distribution="uniform",
    normalize_method="log1p", # plain log1p of RAW counts (no library-size normalisation)
    n_sample_steps=5,
    prior_sampler="zinb",
    zinb_logits=0.1,
    zinb_total_count=1.0,
    zinb_zi_logits=0.0,
    backbone="spatial_transformer",
    hidden_dim=128,
    pairwise_hidden_dim=128,
    n_layers=4,
    dropout=0.2,
    attn_dropout=0.2,
    n_neighbors=8,
    n_heads=4,
    feature_dim=UNI_FEATURE_DIM,
    activation="swiglu",
)

MARKERS = ["ERBB2", "GRB7", "FASN", "GNAS", "ESR1", "PGR", "MKI67"]


def cache_dir(tag):
    return os.path.join(CACHE_ROOT, tag)


def run_dir(tag):
    return os.path.join(RUNS_ROOT, tag)
