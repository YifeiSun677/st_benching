"""训练单个 MLP（在缓存好的 768 维特征上，几秒/个）。"""
import numpy as np
import torch
from torch import nn
from .config import DROPOUT, LR, WEIGHT_DECAY, BATCH, EPOCHS
from .p2s_import import MLP_regression_relu_two

def train_one_mlp(X, Y, seed, device, epochs=None):
    """X:(n,768) Y:(n,833) → 训练好的 model（eval 模式）。
    bias_init = 每基因训练均值（官方架构自带的 hook）：未训练时≈均值预测，稳定。"""
    epochs = epochs or EPOCHS
    torch.manual_seed(seed); np.random.seed(seed)
    n_in, n_out = X.shape[1], Y.shape[1]
    bias_init = torch.tensor(Y.mean(0), dtype=torch.float32)
    model = MLP_regression_relu_two(n_inputs=n_in, n_hiddens=n_in, n_outputs=n_out,
                                    dropout=DROPOUT, bias_init=bias_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(Xt, Yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model

@torch.no_grad()
def predict(model, X, device):
    return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
