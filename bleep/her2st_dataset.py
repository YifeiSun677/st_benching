"""
her2st loader replacing BLEEP's Visium-specific CLIPDataset.

Fixes three bugs in the original dataset.py that make it uncrashable-into:
  * `self.is_train` was read in transform() but never set in __init__
    -> AttributeError on the first __getitem__.
  * `TF.to_tensor` already returns (C,H,W); the original then called
    `.permute(2,0,1)` giving (W,C,H) -- 224 "channels" into conv1.
  * `cv2.imread` returns BGR while the ImageNet mean/std it normalises
    with are RGB. We use PIL and stay RGB throughout.

her2st layout assumed (as cloned from almaan/her2st):
    <root>/ST-cnts/<section>.tsv.gz          spots x genes, index like "10x20"
    <root>/ST-spotfiles/<section>_selection.tsv   x y new_x new_y pixel_x pixel_y
    <root>/ST-imgs/<patient>/<section>/*.jpg[.gz]

Expression target: CPM -> natural log1p, matching the normalisation Wang
et al. 2025 apply to most benchmarked methods. NO Harmony -- BLEEP's own
preprocessing batch-corrects across all four slices including the held-out
one, which leaks test-slice information into both target and reference.
"""
import glob
import gzip
import io
import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # her2st slides exceed the decompression-bomb guard

PATCH = 224
HALF = PATCH // 2


# --------------------------------------------------------------------------
# panel + section helpers
# --------------------------------------------------------------------------
def load_panel(panel_path):
    """One gene identifier per line. Order is fixed and load-bearing:
    it defines the column order of every prediction matrix."""
    with open(panel_path) as fh:
        genes = [ln.strip() for ln in fh if ln.strip()]
    if len(set(genes)) != len(genes):
        raise ValueError(f"{panel_path} contains duplicate gene ids")
    return genes


def sections_for_patient(root, patient):
    """Discover this patient's sections from ST-cnts rather than hardcoding."""
    hits = glob.glob(os.path.join(root, "ST-cnts", f"{patient}[0-9].tsv.gz"))
    return sorted(os.path.basename(p).split(".")[0] for p in hits)


def find_image(root, section):
    patient = section[0]
    pattern = os.path.join(root, "ST-imgs", patient, section, "*.jpg*")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no image under {pattern}")
    return hits[0]


def read_image(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            img = Image.open(io.BytesIO(fh.read())).convert("RGB")
    else:
        img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)  # (H, W, 3)


def read_counts(root, section):
    path = os.path.join(root, "ST-cnts", f"{section}.tsv.gz")
    return pd.read_csv(path, sep="\t", index_col=0)


def read_spots(root, section):
    """Return spot table indexed by the '<x>x<y>' key used in ST-cnts."""
    path = os.path.join(root, "ST-spotfiles", f"{section}_selection.tsv")
    df = pd.read_csv(path, sep="\t")
    xi = np.around(df["x"].values).astype(int)
    yi = np.around(df["y"].values).astype(int)
    df = df.assign(spot_id=[f"{a}x{b}" for a, b in zip(xi, yi)])
    return df.set_index("spot_id")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def cpm_log1p(counts):
    """counts: (n_spots, n_genes) raw. -> CPM, natural log1p."""
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(counts / totals * 1e6)


def align_to_panel(cnt_df, panel):
    """Reindex to the panel, ZERO-FILLING genes absent from this section.

    her2st ST-cnts only carries genes detected in that section, so sparse
    panel genes drop out of individual sections. Zero-fill keeps the matrix
    width fixed at len(panel) across every section and fold -- the same rule
    used for the other models in this benchmark.
    """
    present = [g for g in panel if g in cnt_df.columns]
    out = pd.DataFrame(0.0, index=cnt_df.index, columns=panel)
    out.loc[:, present] = cnt_df.loc[:, present].astype(float).values
    return out, len(present)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class Her2stSection:
    """One section's aligned image patches + panel expression, held in RAM."""

    def __init__(self, root, section, panel, verbose=True):
        self.section = section
        self.panel = panel

        cnt = read_counts(root, section)
        spots = read_spots(root, section)

        shared = [s for s in cnt.index if s in spots.index]
        if not shared:
            raise RuntimeError(f"{section}: no spot ids shared between "
                               f"ST-cnts and ST-spotfiles")
        cnt = cnt.loc[shared]
        spots = spots.loc[shared]

        aligned, n_present = align_to_panel(cnt, panel)
        self.expression = cpm_log1p(aligned.values.astype(np.float32)).astype(np.float32)
        self.spot_ids = list(shared)
        self.cx = spots["pixel_x"].values.astype(float)   # column
        self.cy = spots["pixel_y"].values.astype(float)   # row

        self.image = read_image(find_image(root, section))
        self.n_padded = self._count_padded()

        if verbose:
            print(f"  {section}: {len(shared)} spots, image "
                  f"{self.image.shape[1]}x{self.image.shape[0]}, "
                  f"{n_present}/{len(panel)} panel genes present, "
                  f"{self.n_padded} patches need edge padding")

    def _count_padded(self):
        h, w = self.image.shape[:2]
        bad = ((self.cx - HALF < 0) | (self.cx + HALF > w) |
               (self.cy - HALF < 0) | (self.cy + HALF > h))
        return int(bad.sum())

    def patch(self, i):
        """224x224 RGB crop centred on the spot, white-padded at edges."""
        h, w = self.image.shape[:2]
        x0, y0 = int(round(self.cx[i])) - HALF, int(round(self.cy[i])) - HALF
        x1, y1 = x0 + PATCH, y0 + PATCH
        out = np.full((PATCH, PATCH, 3), 255, dtype=np.uint8)
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x1, w), min(y1, h)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                self.image[sy0:sy1, sx0:sx1]
        return out

    def __len__(self):
        return len(self.spot_ids)


class Her2stCLIPDataset(torch.utils.data.Dataset):
    """Concatenation of sections, exposing BLEEP's expected batch keys."""

    def __init__(self, root, sections, panel, is_train, verbose=True):
        self.is_train = is_train
        self.panel = panel
        if verbose:
            print(f"Loading sections {sections} (is_train={is_train})")
        self.sections = [Her2stSection(root, s, panel, verbose) for s in sections]
        self.index = [(si, i) for si, sec in enumerate(self.sections)
                      for i in range(len(sec))]

    def expression_matrix(self):
        return np.concatenate([s.expression for s in self.sections], axis=0)

    def spot_keys(self):
        return [f"{self.sections[si].section}:{self.sections[si].spot_ids[i]}"
                for si, i in self.index]

    def transform(self, arr):
        image = Image.fromarray(arr)
        if self.is_train:
            # BLEEP's augmentation: random h/v flip + 90-degree rotation
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.choice([180, 90, 0, -90]))
        image = TF.to_tensor(image)  # (3, 224, 224), already channel-first
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        return image

    def __getitem__(self, idx):
        si, i = self.index[idx]
        sec = self.sections[si]
        return {
            "image": self.transform(sec.patch(i)).float(),
            "reduced_expression": torch.from_numpy(sec.expression[i]).float(),
        }

    def __len__(self):
        return len(self.index)
