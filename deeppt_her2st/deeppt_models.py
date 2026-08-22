"""
DeepPT components (iii) and (iv).

  (iii) autoencoder  2048 -> 512 -> 2048
  (iv)  predictor    512  -> hidden -> n_genes

!! The exact layer widths, activations, dropout, optimiser and LR below are a
!! faithful-in-spirit reconstruction, NOT transplanted from the Zenodo code
!! (that archive blocks automated download). After you unzip 12AE.zip and
!! 13DeepPT_train.zip, open 1main_AE.py and 1main_train.py and OVERWRITE the
!! defaults here with the real ones. See INSPECT.md for the checklist.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AE(nn.Module):
    """Compress 2048 pretrained ResNet features to 512."""

    def __init__(self, d_in: int = 2048, d_code: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_in, d_code), nn.ReLU())
        self.decoder = nn.Linear(d_code, d_in)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)


class Predictor(nn.Module):
    """AE code -> gene expression."""

    def __init__(self, d_code: int = 512, d_hidden: int = 512,
                 n_genes: int = 833, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_code, d_hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, n_genes),
        )

    def forward(self, x):
        return self.net(x)
