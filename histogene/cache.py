"""One-off caches.

her2st images are ~9300x9900 px. The original ViT_HER2ST holds every training
section in RAM as a float32 tensor (~1.1 GB each x ~31 sections = ~34 GB) and
re-decodes the JPEGs at the start of every fold. Caching 112x112 uint8 patches
once costs ~0.5 GB on disk and makes every fold start in seconds.

Layout under CACHE_DIR:
    patches/<section>.npy      uint8  (n_spots, 112, 112, 3)   HWC, as cropped
    patches/<section>.npz      spot ids + array coords + pixel coords
    expr/<panel_hash>/<section>.npy   float32 (n_spots, n_genes)
    expr/<panel_hash>/panel.txt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from . import config as C
from . import her2st

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def patch_dir() -> Path:
    return C.CACHE_DIR / "patches"


def _crop(arr: np.ndarray, px: int, py: int, r: int) -> tuple[np.ndarray, bool]:
    """Zero-padded crop, returns (patch HWC, was_padded)."""
    H, W = arr.shape[:2]
    y0, y1, x0, x1 = py - r, py + r, px - r, px + r
    if y0 >= 0 and x0 >= 0 and y1 <= H and x1 <= W:
        return arr[y0:y1, x0:x1, :], False
    out = np.zeros((2 * r, 2 * r, 3), dtype=arr.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, H), min(x1, W)
    out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0, :] = arr[sy0:sy1, sx0:sx1, :]
    return out, True


def build_patch_cache(sections: list[str] | None = None, force: bool = False) -> None:
    r = C.PATCH_R
    d = patch_dir()
    d.mkdir(parents=True, exist_ok=True)
    sections = sections or her2st.section_names()
    report = []
    for name in sections:
        npy, npz = d / f"{name}.npy", d / f"{name}.npz"
        if npy.exists() and npz.exists() and not force:
            print(f"[{name}] cached, skipping")
            continue
        meta = her2st.read_meta(name)
        px = np.floor(meta["pixel_x"].values).astype(int)
        py = np.floor(meta["pixel_y"].values).astype(int)
        ax = np.around(meta["x"].values).astype(int)
        ay = np.around(meta["y"].values).astype(int)

        img = np.asarray(Image.open(her2st.image_path(name)).convert("RGB"))
        patches = np.zeros((len(meta), 2 * r, 2 * r, 3), dtype=np.uint8)
        n_pad = 0
        for i in range(len(meta)):
            p, padded = _crop(img, px[i], py[i], r)
            patches[i] = p
            n_pad += int(padded)
        del img

        np.save(npy, patches)
        np.savez(npz,
                 spot_id=np.array(meta.index, dtype=object),
                 array_x=ax, array_y=ay, pixel_x=px, pixel_y=py)
        report.append({"section": name, "n_spots": int(len(meta)),
                       "n_padded": n_pad,
                       "max_array_x": int(ax.max()), "max_array_y": int(ay.max())})
        print(f"[{name}] {len(meta):5d} spots, {n_pad} edge-padded, "
              f"array max ({ax.max()},{ay.max()})")
    if report:
        (C.CACHE_DIR / "patch_cache_report.json").write_text(json.dumps(report, indent=2))


# ------------------------------------------------------------ expression -----
def panel_hash(panel: list[str]) -> str:
    return hashlib.sha1("\n".join(panel).encode()).hexdigest()[:10]


def expr_dir(panel: list[str]) -> Path:
    return C.CACHE_DIR / "expr" / panel_hash(panel)


def build_expr_cache(panel: list[str], sections: list[str] | None = None,
                     force: bool = False) -> None:
    d = expr_dir(panel)
    d.mkdir(parents=True, exist_ok=True)
    (d / "panel.txt").write_text("\n".join(panel) + "\n")
    for name in sections or her2st.section_names():
        f = d / f"{name}.npy"
        if f.exists() and not force:
            continue
        meta = her2st.read_meta(name)
        np.save(f, her2st.expression(meta, panel))
        print(f"[{name}] expression cached {len(meta)} x {len(panel)}")


def load_expr(panel: list[str], name: str) -> np.ndarray:
    f = expr_dir(panel) / f"{name}.npy"
    if not f.exists():
        build_expr_cache(panel, [name])
    return np.load(f)


def load_patches(name: str) -> np.ndarray:
    return np.load(patch_dir() / f"{name}.npy", mmap_mode="r")


def load_coords(name: str):
    z = np.load(patch_dir() / f"{name}.npz", allow_pickle=True)
    return z
