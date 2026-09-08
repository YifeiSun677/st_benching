"""
Build a one-off 112x112 uint8 patch cache for her2st, reproducing the EXACT
crop ViT_HER2ST uses in the TCGN repo, so numbers match the upstream pipeline.

Upstream crop (dataset.py __getitem__):
    im = img_tensor.permute(1, 0, 2)                 # (H,W,3) -> (W,H,3)  i.e. (x,y)
    patch = im[(x-r):(x+r), (y-r):(y+r), :]          # x,y = floor(pixel_x,pixel_y)
We reproduce it verbatim with numpy: t = arr.transpose(1,0,2); t[x-r:x+r, y-r:y+r].
That means each stored patch is the spatial transpose of the "natural" crop; we
keep it because TCGN trains AND tests with this convention, so it is internally
consistent (unlike the Hist2ST axis bug, where cache and crop disagreed).

Layout: one row per spot in a global memmap
    patches.uint8   shape (N, 112, 112, 3)
    index.npz       section (str[N]), spot_id (str[N]), center (int[N,2])
Sections are decoded once each (~275 MB each); ~0.5 GB total, ~15-30 min.
"""
import os
import numpy as np
from PIL import Image, ImageFile

import config as C
import her2_data as H

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

PATCH = 2 * C.R  # 112


def _crop_transposed(arr, cx, cy, r):
    """arr: (H,W,3) uint8. Returns (2r,2r,3) using upstream's transpose convention.
    Edge spots are zero-padded so the patch is always (2r,2r,3)."""
    t = np.transpose(arr, (1, 0, 2))          # (W,H,3) == (x,y,3)
    W, Hh = t.shape[0], t.shape[1]
    out = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(Hh, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[(sx0 - x0):(sx1 - x0), (sy0 - y0):(sy1 - y0), :] = t[sx0:sx1, sy0:sy1, :]
    return out


def build(sections=None, force=False):
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    patches_path = os.path.join(C.CACHE_DIR, "patches.uint8.npy")
    index_path = os.path.join(C.CACHE_DIR, "index.npz")
    if os.path.exists(patches_path) and os.path.exists(index_path) and not force:
        print("[patch_cache] already built at", C.CACHE_DIR, "(use force=True to rebuild)")
        return patches_path, index_path

    panel = H.load_panel()
    sections = sections or H.list_sections()

    # first pass: count spots
    counts = {}
    for name in sections:
        _, centers, _ = H.section_targets(name, panel)
        counts[name] = len(centers)
    N = sum(counts.values())
    print("[patch_cache] %d sections, %d spots -> %.2f GB"
          % (len(sections), N, N * PATCH * PATCH * 3 / 1e9))

    patches = np.lib.format.open_memmap(
        patches_path, mode="w+", dtype=np.uint8, shape=(N, PATCH, PATCH, 3))
    sec_arr = np.empty(N, dtype=object)
    sid_arr = np.empty(N, dtype=object)
    cen_arr = np.zeros((N, 2), dtype=np.int32)

    row = 0
    for name in sections:
        _, centers, spot_ids = H.section_targets(name, panel)
        arr = np.asarray(Image.open(H.get_img_path(name)).convert("RGB"), dtype=np.uint8)
        for i in range(len(centers)):
            cx, cy = int(centers[i][0]), int(centers[i][1])
            patches[row] = _crop_transposed(arr, cx, cy, C.R)
            sec_arr[row] = name
            sid_arr[row] = str(spot_ids[i])
            cen_arr[row] = (cx, cy)
            row += 1
        del arr
        print("  cached %s (%d spots)" % (name, len(centers)))
    patches.flush()
    np.savez(index_path, section=sec_arr, spot_id=sid_arr, center=cen_arr)
    print("[patch_cache] done:", patches_path)
    return patches_path, index_path


def load():
    patches = np.load(os.path.join(C.CACHE_DIR, "patches.uint8.npy"), mmap_mode="r")
    idx = np.load(os.path.join(C.CACHE_DIR, "index.npz"), allow_pickle=True)
    return patches, idx["section"], idx["spot_id"], idx["center"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sections", nargs="*", default=None)
    args = ap.parse_args()
    build(sections=args.sections, force=args.force)
