#!/usr/bin/env python
"""
check_colour.py - is a patch cache colour or grayscale?

BT.601 gray (bleep/gray.py) writes R == G == B in every pixel, so a gray
cache has exactly 0 pixels where the channels differ. A colour H&E cache
has almost every pixel differing.

Usage
  python check_colour.py                      # auto-find *.npy caches under /workspace
  python check_colour.py path/to/cache.npy    # one or more .npy files
  python check_colour.py raw.dat --shape 13620,112,112,3 --dtype uint8   # raw memmap
"""
import argparse
import os
import sys

import numpy as np

N_SAMPLE = 500


def find_caches(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"runs", ".git", "__pycache__"}]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if fn.endswith(".npy") and any(k in p.lower() for k in ("cache", "patch")):
                hits.append(p)
    return sorted(hits)


def channel_diff_fraction(x):
    """x: int16 array. Returns dict layout -> fraction of pixels with differing channels."""
    out = {}
    if x.ndim == 4 and x.shape[-1] == 3:
        out["HWC"] = x
    if x.ndim == 4 and x.shape[1] == 3:
        out["CHW"] = np.moveaxis(x, 1, -1)
    if x.ndim == 2 and x.shape[1] % 3 == 0:          # flattened patches
        s = int(round((x.shape[1] // 3) ** 0.5))
        if s * s * 3 == x.shape[1]:
            out["flat-HWC"] = x.reshape(-1, s, s, 3)
            out["flat-CHW"] = np.moveaxis(x.reshape(-1, 3, s, s), 1, -1)
    res = {}
    for k, v in out.items():
        r, g, b = v[..., 0], v[..., 1], v[..., 2]
        res[k] = float(((r != g) | (g != b)).mean())
    return res


def check(path, shape=None, dtype="uint8"):
    try:
        if shape:
            a = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        else:
            a = np.load(path, mmap_mode="r")
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None
    print(f"\n{path}\n  shape {a.shape}  dtype {a.dtype}")
    n = a.shape[0]
    idx = np.sort(np.random.default_rng(0).choice(n, size=min(N_SAMPLE, n), replace=False))
    x = np.asarray(a[idx]).astype(np.int16)
    res = channel_diff_fraction(x)
    if not res:
        print("  [skip] not an image-patch layout")
        return None
    for k, v in res.items():
        print(f"  layout {k:9s} fraction of pixels with R/G/B differing: {v:.4f}")
    # gray shows 0 in the true layout; colour is > 0 in every layout
    verdict = "GRAY" if min(res.values()) == 0.0 else "COLOUR"
    print(f"  VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--shape", default=None, help="for raw memmaps, e.g. 13620,112,112,3")
    ap.add_argument("--dtype", default="uint8")
    args = ap.parse_args()
    shape = tuple(int(s) for s in args.shape.split(",")) if args.shape else None

    paths = args.paths or find_caches(args.root)
    if not paths:
        sys.exit(f"No *cache*/*patch* .npy files found under {args.root}; pass the path explicitly.")
    verdicts = {p: check(p, shape, args.dtype) for p in paths}
    print("\nSUMMARY")
    for p, v in verdicts.items():
        if v:
            print(f"  {v:6s}  {p}")


if __name__ == "__main__":
    main()
