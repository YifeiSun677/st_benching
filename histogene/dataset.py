"""Section-level dataset reproducing the upstream ViT_HER2ST tensor layout.

One dataset item = one whole section:
    patches   float32 [n_spots, 3*112*112]   raw 0-255, NOT divided by 255
    positions int64   [n_spots, 2]           array coords, feed nn.Embedding
    exps      float32 [n_spots, n_genes]     log10(CP10K + 1)
    centers   float32 [n_spots, 2]           pixel coords (test split only)

Layout note: the repo permutes the full image to (x, y, c) before cropping, so
its flattened patch is the transpose of a normal HWC crop. We reproduce that
here with .transpose(0, 2, 1, 3) so weights and this loader stay interchangeable
with the upstream code.

Grayscale (colour ablation)
---------------------------
``gray=True`` converts the uint8 patches to BT.601 luma replicated on 3 channels
using the canonical ``bleep/gray.py`` transform (see gray_bridge.py). It is
applied immediately after the cache read, before transpose / float / flatten:
  * the 112x112 disk cache is not rebuilt or modified,
  * the item is still (n, 37632), so Linear(37632 -> 1024) is unchanged,
  * the scale is still 0-255.
``gray=False`` (default) is byte-for-byte the previous behaviour, so train.py,
preflight.py and overfit_probe.py are unaffected.
"""
from __future__ import annotations

import numpy as np
import torch

from . import cache, config as C


class HER2STSections(torch.utils.data.Dataset):
    def __init__(self, sections: list[str], panel: list[str], train: bool = True,
                 gray: bool = False):
        self.sections = list(sections)
        self.panel = panel
        self.train = train
        self.gray = bool(gray)
        if self.gray:
            # import lazily so colour runs never depend on bleep/gray.py
            from .gray_bridge import assert_canonical, gray_section
            assert_canonical()
            self._gray_fn = gray_section
        else:
            self._gray_fn = None

        self.patches = {s: cache.load_patches(s) for s in self.sections}
        self.exprs = {s: cache.load_expr(panel, s) for s in self.sections}
        self.coords = {s: cache.load_coords(s) for s in self.sections}
        for s in self.sections:
            n_p, n_e = self.patches[s].shape[0], self.exprs[s].shape[0]
            if n_p != n_e:
                raise RuntimeError(
                    f"{s}: patch cache has {n_p} spots but expression has {n_e}. "
                    "Rebuild the caches (python -m histogene.build_cache --force).")
        mx = max(int(self.coords[s]["array_x"].max()) for s in self.sections)
        my = max(int(self.coords[s]["array_y"].max()) for s in self.sections)
        if max(mx, my) >= C.N_POS:
            raise RuntimeError(f"array coord {max(mx, my)} >= n_pos {C.N_POS}")

    def __len__(self) -> int:
        return len(self.sections)

    def n_spots(self) -> int:
        return sum(self.patches[s].shape[0] for s in self.sections)

    def raw_patches(self, i: int) -> np.ndarray:
        """uint8 (n, 112, 112, 3) exactly as the model's input is built from,
        i.e. AFTER the optional grayscale step and BEFORE transpose."""
        p = np.asarray(self.patches[self.sections[i]])      # (n, 112, 112, 3) uint8
        if self._gray_fn is not None:
            p = self._gray_fn(p)                            # same shape/dtype, R==G==B
        return p

    def __getitem__(self, i: int):
        s = self.sections[i]
        p = self.raw_patches(i)
        p = p.transpose(0, 2, 1, 3)                         # -> (n, x, y, 3), repo order
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
