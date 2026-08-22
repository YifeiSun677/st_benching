"""
her2st reading conventions, in one place.

Mirrors what HisToGene's ViT_HER2ST and DeepHis2Exp's HER2ST loader do, so the
spot set is identical to your ST-Net / HisToGene runs:

  ST-cnts/<sec>.tsv.gz          rows = spot ids "<x>x<y>", cols = gene symbols
  ST-spotfiles/<sec>_selection.tsv.gz
                                cols = x, y, pixel_x, pixel_y, selected
  ST-imgs/<patient>/<sec>/HE_*.jpg

Join key is the string f"{x}x{y}". Patch is the 224x224 square centred on
(pixel_x, pixel_y), i.e. half-window r = 112.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # her2st JPEGs are ~10k x 10k

R = 112  # half-window -> 224 px patch, same as HisToGene


def list_sections(root: str) -> list[str]:
    """All section ids (A1..H3) present in ST-cnts, sorted."""
    paths = glob.glob(os.path.join(root, "data", "ST-cnts", "*.tsv.gz"))
    secs = [os.path.basename(p).split(".")[0] for p in paths]
    return sorted(secs)


def patient_of(section: str) -> str:
    """'B3' -> 'B'."""
    m = re.match(r"^([A-Z])", section)
    if not m:
        raise ValueError(f"cannot parse patient from section {section!r}")
    return m.group(1)


def image_path(root: str, section: str) -> str:
    pat = patient_of(section)
    hits = glob.glob(os.path.join(root, "data", "ST-imgs", pat, section, "*.jpg"))
    if not hits:
        hits = glob.glob(os.path.join(root, "data", "ST-imgs", pat, section, "*.jpeg"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{section}: expected 1 image, found {hits}")
    return hits[0]


def load_counts(root: str, section: str) -> pd.DataFrame:
    """Raw count matrix, index = spot id, columns = gene symbols."""
    p = os.path.join(root, "data", "ST-cnts", f"{section}.tsv.gz")
    df = pd.read_csv(p, sep="\t", index_col=0, compression="gzip")
    df.index = df.index.astype(str)
    return df


def load_spots(root: str, section: str) -> pd.DataFrame:
    """Selection file, indexed by the same '<x>x<y>' key as the count matrix."""
    p = os.path.join(root, "data", "ST-spotfiles", f"{section}_selection.tsv.gz")
    df = pd.read_csv(p, sep="\t", compression="gzip")
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["x", "y", "pixel_x", "pixel_y"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise KeyError(f"{section}: selection file missing {missing}; has {list(df.columns)}")
    out = pd.DataFrame({
        "x": df[cols["x"]].astype(float).round().astype(int),
        "y": df[cols["y"]].astype(float).round().astype(int),
        "pixel_x": df[cols["pixel_x"]].astype(float),
        "pixel_y": df[cols["pixel_y"]].astype(float),
    })
    if "selected" in cols:
        out = out[df[cols["selected"]].astype(int) == 1].reset_index(drop=True)
    out.index = out["x"].astype(str) + "x" + out["y"].astype(str)
    return out


def aligned_spots(root: str, section: str) -> pd.DataFrame:
    """Spots present in BOTH the count matrix and the selection file, in a
    fixed (sorted) order. This is the canonical spot set for the section."""
    cnt = load_counts(root, section)
    pos = load_spots(root, section)
    keep = sorted(set(cnt.index) & set(pos.index))
    if not keep:
        raise RuntimeError(f"{section}: no spot ids shared between cnts and spotfile")
    return pos.loc[keep]


def crop_patches(img: Image.Image, pos: pd.DataFrame, r: int = R) -> np.ndarray:
    """[n_spots, 2r, 2r, 3] uint8. Spots whose window falls off the slide edge
    are zero-padded rather than dropped, so the spot set never changes."""
    W, H = img.size
    arr = np.asarray(img)  # H, W, 3
    n = len(pos)
    out = np.zeros((n, 2 * r, 2 * r, 3), dtype=np.uint8)
    for i, (px, py) in enumerate(zip(pos["pixel_x"].values, pos["pixel_y"].values)):
        cx, cy = int(round(px)), int(round(py))
        x0, x1 = cx - r, cx + r
        y0, y1 = cy - r, cy + r
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x1), min(H, y1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        out[i, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out
