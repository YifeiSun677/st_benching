"""读特征缓存、拼训练/测试矩阵、做 target 变换。"""
import numpy as np
from .config import FEATURE_DIR, TARGET_TRANSFORM

def transform_counts(C):
    """C: 原始 counts (n,833) → 非负目标（匹配 MLP 输出 ReLU）。"""
    if TARGET_TRANSFORM == "lognorm":
        lib = C.sum(1, keepdims=True); lib[lib == 0] = 1.0
        pos = C.sum(1)[C.sum(1) > 0]
        med = np.median(pos) if pos.size else 1.0
        return np.log1p(C / lib * med)            # 中位文库归一化 + 自然 log1p
    if TARGET_TRANSFORM == "log10_cp10k":
        lib = C.sum(1, keepdims=True); lib[lib == 0] = 1.0
        return np.log10(C / lib * 1e4 + 1.0)      # 想和 HisToGene/Hist2ST 口径一致时用
    raise ValueError(TARGET_TRANSFORM)

def load_section(section):
    d = np.load(FEATURE_DIR / f"{section}.npz", allow_pickle=True)
    return d

def build_xy(sections):
    """把若干 section 拼成 (X 特征, Y 变换后目标)。"""
    Xs, Ys = [], []
    for s in sections:
        d = load_section(s)
        Xs.append(d["feat"].astype(np.float32))
        Ys.append(transform_counts(d["counts833"].astype(np.float32)))
    return np.concatenate(Xs), np.concatenate(Ys)
