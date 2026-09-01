import sys, torch
import torch.nn.functional as F
from . import config as C, her2st
from .dataset import HER2STSections

sys.path.insert(0, str(C.HISTOGENE_REPO))
from vis_model import HisToGene

panel = her2st.load_panel(str(C.PANEL_FILE))
ds = HER2STSections(["B1"], panel, train=True)
patches, positions, exps = ds[0]

print("patches", tuple(patches.shape),
      f"min={patches.min():.2f} max={patches.max():.2f} "
      f"mean={patches.mean():.2f} std={patches.std():.2f}")
print("exps   ", tuple(exps.shape), f"mean={exps.mean():.4f} std={exps.std():.4f}")
baseline = ((exps - exps.mean(0)) ** 2).mean().item()
print(f"baseline loss (target variance) = {baseline:.4f}\n")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=C.N_LAYERS,
                  n_genes=len(panel), dim=C.DIM, learning_rate=1e-4,
                  dropout=0.0, n_pos=C.N_POS).to(dev)          # dropout 关掉
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

P, Q, Y = (patches.unsqueeze(0).to(dev), positions.unsqueeze(0).to(dev),
           exps.unsqueeze(0).to(dev))
model.train()
for step in range(3000):
    opt.zero_grad()
    loss = F.mse_loss(model(P, Q), Y)
    loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:5d}  loss {loss.item():.4f}", flush=True)