"""
DeepPT components (iii) and (iv) -- now transplanted from the released code
(12AE/model_AE.py, 13DeepPT_train/model_MLP.py), not reconstructed.

Two things differ from the obvious guess, both verified against the source:

  * the AE decoder ends in ReLU. Post-avgpool ResNet50 features are ReLU
    outputs and therefore non-negative, so the decoder can represent them
    exactly. This is why the input features must NOT be standardised --
    z-scored features have negatives a ReLU decoder can never reconstruct.

  * the predictor has NO ReLU. The original commented it out deliberately
    ("2020.03.26: for positive gene expression"), leaving
    Linear -> Dropout -> Linear, i.e. a regularised LINEAR regression on the
    AE codes. Do not "fix" this -- it is the published model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AE(nn.Module):
    """12AE/model_AE.py :: AutoEncoder. 2048 -> 512 -> 2048, ReLU on both ends."""

    def __init__(self, d_in: int = 2048, d_code: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d_code), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(d_code, d_in), nn.ReLU())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)


class Predictor(nn.Module):
    """13DeepPT_train/model_MLP.py :: MLP_regression.

    Linear -> Dropout -> Linear. No activation between the layers: the ReLU
    is commented out in the original. Output-layer bias is initialised to the
    per-gene TRAINING mean, so the model starts at the mean-expression
    baseline and learns the residual.

    The original's `torch.mean(x, dim=0)` tile-averaging is dropped: at
    spot level there is one tile per spot, so there is nothing to average.
    """

    def __init__(self, d_code: int = 512, d_hidden: int = 512,
                 n_genes: int = 833, p_drop: float = 0.2,
                 bias_init=None):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(d_code, d_hidden),
            nn.Dropout(p_drop),
        )
        self.layer1 = nn.Linear(d_hidden, n_genes)
        if bias_init is not None:
            with torch.no_grad():
                self.layer1.bias.copy_(
                    torch.as_tensor(bias_init, dtype=torch.float32))

    def forward(self, x):
        return self.layer1(self.layer0(x))
