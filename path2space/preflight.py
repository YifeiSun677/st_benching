"""开跑前自检：权重、panel、CTransPath 加载、MLP 前向。"""
import torch
from .config import CTRANSPATH, PANEL_FILE, FEATURE_DIR
from .p2s_import import CTransPathExtractor, MLP_regression_relu_two, init_random_seed

def main():
    init_random_seed(0)
    assert CTRANSPATH.exists(), f"缺 CTransPath 权重：{CTRANSPATH}"
    panel = [g for g in PANEL_FILE.read_text().splitlines() if g.strip()]
    assert len(panel) == 833, len(panel)
    CTransPathExtractor(str(CTRANSPATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = MLP_regression_relu_two(768, 768, 833, dropout=0.2,
                                bias_init=torch.zeros(833)).to(device)
    y = m(torch.randn(4, 768, device=device))
    assert y.shape == (4, 833) and (y >= 0).all(), "MLP 前向异常（输出应非负）"
    man = FEATURE_DIR / "manifest.json"
    print("特征缓存:", ("已就绪 " + str(man)) if man.exists()
          else "尚未构建（先跑 build_features）")
    print("preflight OK ✔")

if __name__ == "__main__":
    main()
