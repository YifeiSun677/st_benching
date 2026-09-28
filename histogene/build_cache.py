#!/usr/bin/env python
"""
Pre-crop every her2st spot patch once and cache it as uint8.

HisToGene's ViT_HER2ST re-crops every patch from the full WSI on every epoch,
and holds all training WSIs in RAM as float32 (~30-40 GB). Because
ViT_HER2ST.__getitem__ never applies self.transforms, the patches are byte
identical on every epoch, so this work is pure waste.

This script does the cropping once. Output is ~510 MB total and every fold
reuses it. Raw counts are cached alongside the patches so switching gene panel
(250 / 785 / 833) does not require re-cropping.

Cache layout, one .npz per section:
    patches  uint8   (n_spots, 112*112*3)   flattened, matches ViT_HER2ST layout
    loc      int64   (n_spots, 2)           array coords -> position embedding
    centers  int64   (n_spots, 2)           pixel coords
    genes    <U      (n_genes_section,)     this section's detected genes
    counts   float32 (n_spots, n_genes_section)

Usage:
    python build_cache.py --her2st data/her2st --out cache
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

R = 56  # ViT_HER2ST uses self.r = 224 // 4 -> 112x112 patches


def get_pos(pos_dir, name):
    df = pd.read_csv(f"{pos_dir}/{name}_selection.tsv", sep="\t")
    x = np.around(df["x"].values).astype(int)
    y = np.around(df["y"].values).astype(int)
    df["id"] = [f"{a}x{b}" for a, b in zip(x, y)]
    return df


def build_section(name, cnt_dir, img_dir, pos_dir, out_dir):
    cnt = pd.read_csv(f"{cnt_dir}/{name}.tsv", sep="\t", index_col=0)
    meta = cnt.join(get_pos(pos_dir, name).set_index("id"))
    meta = meta.dropna(subset=["pixel_x", "pixel_y", "x", "y"])

    img_path = glob.glob(f"{img_dir}/{name[0]}/{name}/*")[0]
    im = np.array(Image.open(img_path))  # (H, W, 3) uint8
    H, W = im.shape[0], im.shape[1]

    centers = np.floor(meta[["pixel_x", "pixel_y"]].values).astype(int)
    loc = meta[["x", "y"]].values.astype(np.int64)

    n = len(centers)
    P = np.zeros((n, 2 * R, 2 * R, 3), dtype=np.uint8)
    n_clipped = 0
    for i, (x, y) in enumerate(centers):
        # ViT_HER2ST crops im.permute(1,0,2)[x-r:x+r, y-r:y+r], i.e. dim0 is the
        # pixel_x axis. Equivalent to slicing (row=y, col=x) then transposing the
        # small patch -- much faster than transposing the whole WSI.
        x0, x1 = max(x - R, 0), min(x + R, W)
        y0, y1 = max(y - R, 0), min(y + R, H)
        if (x1 - x0) != 2 * R or (y1 - y0) != 2 * R:
            n_clipped += 1
        P[i, x0 - (x - R):x1 - (x - R), y0 - (y - R):y1 - (y - R)] = \
            im[y0:y1, x0:x1].transpose(1, 0, 2)

    np.savez_compressed(
        f"{out_dir}/{name}.npz",
        patches=P.reshape(n, -1),
        loc=loc,
        centers=centers,
        genes=np.array(cnt.columns, dtype=object),
        counts=cnt.loc[meta.index].values.astype(np.float32),
    )
    del im, P
    return n, cnt.shape[1], (H, W), n_clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2st", default="data/her2st",
                    help="path to the her2st repo root (contains data/ST-cnts)")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--force", action="store_true", help="rebuild sections that already exist")
    a = ap.parse_args()

    cnt_dir = f"{a.her2st}/data/ST-cnts"
    img_dir = f"{a.her2st}/data/ST-imgs"
    pos_dir = f"{a.her2st}/data/ST-spotfiles"
    for d in (cnt_dir, img_dir, pos_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"missing: {d}  (is --her2st pointing at the repo root?)")

    os.makedirs(a.out, exist_ok=True)
    names = sorted(f[:2] for f in os.listdir(cnt_dir) if f.endswith(".tsv"))
    print(f"{len(names)} sections: {' '.join(names)}\n")

    total_spots = 0
    for name in names:
        dest = f"{a.out}/{name}.npz"
        if os.path.exists(dest) and not a.force:
            print(f"{name}  skip (exists)")
            total_spots += int(np.load(dest, allow_pickle=True)["loc"].shape[0])
            continue
        n, ng, (H, W), nc = build_section(name, cnt_dir, img_dir, pos_dir, a.out)
        total_spots += n
        clip = f"  [{nc} spots clipped at border]" if nc else ""
        print(f"{name}  {n:5d} spots  {ng:6d} genes  WSI {W}x{H}{clip}", flush=True)

    size = sum(os.path.getsize(f) for f in glob.glob(f"{a.out}/*.npz")) / 1e9
    print(f"\ndone: {total_spots} spots, cache = {size:.2f} GB in {a.out}/")


if __name__ == "__main__":
    main()
