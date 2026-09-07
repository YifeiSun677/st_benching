"""集中所有可调项。改路径/超参只动这里。"""
from pathlib import Path

# ---- 路径 ----
ST_ROOT      = Path("/workspace/her2st/data")                 # her2st 数据根目录
P2S_UPSTREAM = Path("/workspace/p2s_upstream")                # 官方 companion 克隆
CTRANSPATH   = Path("/workspace/p2s_weights/ctranspath.pth")  # CTransPath 权重
REPO         = Path(__file__).resolve().parents[1]            # st_benching 根
PANEL_FILE   = REPO / "panels" / "panel_833.txt"
FEATURE_DIR  = Path("/workspace/p2s_feature_cache")           # 特征缓存输出（一次性，~42MB）
OUT_DIR      = REPO / "results" / "path2space_lopo_833"       # 打分/预测输出

# ---- patch ----
PATCH_PX = 224          # 每个 spot 裁 224×224，CTransPath 内部再 Resize 到 224

# ---- target 变换（★ 假设项，见 WORKING_PROCESS.md）----
# 'lognorm' = 中位文库大小归一化 + 自然 log1p（非负，匹配 MLP 输出的 ReLU）
# 想和你跨模型统一口径，可改成 log10(CP10K+1) 等——只需改 dataset.py::transform_counts
TARGET_TRANSFORM = "lognorm"

# ---- MLP 训练超参（★ 公开代码没有，给合理默认，见 WORKING_PROCESS.md）----
DROPOUT       = 0.2      # 官方架构固定值（SAME）
LR            = 1e-4
WEIGHT_DECAY  = 0.0
BATCH         = 256
EPOCHS        = 200      # 固定预算、last-epoch 打分（与你 benchmark 规则一致）；先用 probe 定
OPTIMIZER     = "adam"

# ---- 集成（★ 22×7 在 8-patient LOPO 下无法照搬，见 WORKING_PROCESS.md）----
# N_IK：外层子集数。None = 对 7 个训练 patient 做 leave-one-patient-out（=7）
#       设成整数则用该数量的 patient 子集；测试时设 1
# N_IL：内层重复数（不同随机种子 → 集成多样性）；测试时设 1
N_IK = None
N_IL = 7

# ---- 评估 ----
MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]  # 与你其他模型一致
SMOOTH_RADIUS = 2       # Path2Space 的 KDTree 空间平滑（array-coord 上半径2）；作为附加变体报告

SEED = 42

# ---- 测试模式（Step 3）----
TEST_PATIENTS = ["A", "B"]   # 只对这两个 patient 建特征 + 跑 2-fold mini-LOPO
