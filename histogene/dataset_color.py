"""Section-level dataset, byte-for-byte compatible with the original ViT_HER2ST.

One dataset item = one whole section:
    patches   float32 [n_spots, 3*112*112]   raw 0-255, NOT divided by 255
    positions int64   [n_spots, 2]           array coords, feed nn.Embedding
    exps      float32 [n_spots, n_genes]     log10(CP10K + 1)
    centers   float32 [n_spots, 2]           pixel coords (test split only)

Layout note: the repo permutes the full image to (x, y, c) before cropping, so
its flattened patch is the transpose of a normal HWC crop. We reproduce that
here with .transpose(1, 0, 2) so weights and this loader stay interchangeable
with the upstream code.
"""
from __future__ import annotations

import numpy as np
import torch

from . import cache, config as C


class HER2STSections(torch.utils.data.Dataset):
    def __init__(self, sections: list[str], panel: list[str], train: bool = True):
        self.sections = list(sections)
        self.panel = panel
        self.train = train
        self.patches = {s: cache.load_patches(s) for s in self.sections}
        self.exprs = {s: cache.load_expr(panel, s) for s in self.sections}
        self.coords = {s: cache.load_coords(s) for s in self.sections}
        for s in self.sections:
            n_p, n_e = self.patches[s].shape[0], self.exprs[s].shape[0]
            if n_p != n_e:
                raise RuntimeError(
                    f"{s}: patch cache has {n_p} spots but expression has {n_e}. "
                    "Rebuild the caches (scripts/01_build_cache.py --force)."
                )
        mx = max(int(self.coords[s]["array_x"].max()) for s in self.sections)
        my = max(int(self.coords[s]["array_y"].max()) for s in self.sections)
        if max(mx, my) >= C.N_POS:
            raise RuntimeError(f"array coord {max(mx, my)} >= n_pos {C.N_POS}")

    def __len__(self) -> int:
        return len(self.sections)

    def n_spots(self) -> int:
        return sum(self.patches[s].shape[0] for s in self.sections)

    def __getitem__(self, i: int):
        s = self.sections[i]
        p = np.asarray(self.patches[s])                    # (n, 112, 112, 3) uint8
        p = p.transpose(0, 2, 1, 3)                        # -> (n, x, y, 3), repo order
        patches = torch.from_numpy(np.ascontiguousarray(p)).float().flatten(1)
        z = self.coords[s]
        positions = torch.from_numpy(
            np.stack([z["array_x"], z["array_y"]], axis=1)).long()
        exps = torch.from_numpy(self.exprs[s]).float()
        if self.train:
            return patches, positions, exps
        centers = torch.from_numpy(
            np.stack([z["pixel_x"], z["pixel_y"]], axis=1)).float()
        return patches, positions, exps, centers
