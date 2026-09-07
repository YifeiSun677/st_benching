"""把官方 companion 的冻结模块 import 进来（只读、不改，逐字复用）。

注意：官方 companion 的包名也叫 `path2space`，与本端口的 `path2space/` 目录同名，
直接 `from path2space.features import ...` 会解析到本端口目录（没有 features.py）而报错。
所以这里用 importlib 把 companion 以别名 `p2s_companion` 加载，彻底避开命名冲突；
其内部的相对 import（如 `from .frozen.ctrans import CTransPath`）会正确解析到别名下。
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from .config import P2S_UPSTREAM

_PKG = Path(P2S_UPSTREAM) / "ge_model" / "path2space"
assert _PKG.exists(), f"找不到官方 companion 包：{_PKG}（先 clone path2space-companion）"

if "p2s_companion" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "p2s_companion", _PKG / "__init__.py",
        submodule_search_locations=[str(_PKG)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["p2s_companion"] = _mod
    _spec.loader.exec_module(_mod)      # 触发 companion 的 __init__（需已装 cv2/spams/torch）

# 通过别名拉子模块（相对 import 在内部解析为 p2s_companion.*）
_features = importlib.import_module("p2s_companion.features")
_colornorm = importlib.import_module("p2s_companion.frozen.utils_color_norm")
_prep = importlib.import_module("p2s_companion.frozen.utils_preprocessing")
_mlp = importlib.import_module("p2s_companion.model_mlp")
_smooth = importlib.import_module("p2s_companion.smoothing")

CTransPathExtractor = _features.CTransPathExtractor
CTRANSPATH_FEATURE_DIM = _features.CTRANSPATH_FEATURE_DIM
macenko_normalizer = _colornorm.macenko_normalizer
evaluate_tile = _prep.evaluate_tile
init_random_seed = _prep.init_random_seed
MLP_regression_relu_two = _mlp.MLP_regression_relu_two
smooth_genes_kdtree = _smooth.smooth_genes_kdtree

assert CTRANSPATH_FEATURE_DIM == 768
