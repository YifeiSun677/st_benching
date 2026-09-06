"""唯一处理 sys.path 的地方：把官方冻结模块 import 进来（只读、不改，逐字复用）。"""
import sys
from .config import P2S_UPSTREAM

_GE = str(P2S_UPSTREAM / "ge_model")
if _GE not in sys.path:
    sys.path.insert(0, _GE)

# 官方原件（frozen）—— 与论文预测数值对齐的关键
from path2space.features import CTransPathExtractor, CTRANSPATH_FEATURE_DIM   # noqa: E402
from path2space.frozen.utils_color_norm import macenko_normalizer             # noqa: E402
from path2space.frozen.utils_preprocessing import evaluate_tile, init_random_seed  # noqa: E402
from path2space.model_mlp import MLP_regression_relu_two                       # noqa: E402
from path2space.smoothing import smooth_genes_kdtree                           # noqa: E402

assert CTRANSPATH_FEATURE_DIM == 768
